import os
import torch
from torchvision import transforms, models
from PIL import Image
import argparse
import timm
import numpy as np
import pandas as pd
import torch.nn as nn
import random
from sklearn.metrics import roc_auc_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from denoiser import get_denoiser
denoiser = get_denoiser(sigma=1, cuda=device)

def img2noise(x):
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    with torch.no_grad():
        residual = denoiser.network(x)/256.0  # [B, C, H, W], same shape as x
    return residual

def get_log_magnitude(x):  # x: (B, C, H, W)
    x_fft = torch.fft.fft2(x)
    x_mag = torch.abs(x_fft)
    x_log_mag = torch.log1p(x_mag)
    return x_log_mag

class FFTWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        x = get_log_magnitude(x)
        return self.base(x)

def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

def get_image_subdirs(root_dir):
    image_subdirs = []
    for subdir, _, files in os.walk(root_dir):
        if any(is_image_file(f) for f in files):
            image_subdirs.append(subdir)
    
    image_subdirs.sort()
    return image_subdirs

def create_class_mapping(dataset_dirs):
    class_names = [os.path.basename(subdir) for subdir in dataset_dirs]
    
    name_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    id_to_name = {i + 1: name for i, name in enumerate(class_names)}
    id_to_path = {i + 1: path for i, path in enumerate(dataset_dirs)}
    
    return name_to_id, id_to_name, id_to_path

class FeaturePairClassifier(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, hidden_dim),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        logits = self.fc(x)
        return logits.squeeze(1)

def load_feature_model(model_path, model_type='xception', latent_size=128):
    if model_type == 'res50':
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, latent_size)
    elif model_type == 'effb0':
        model = models.efficientnet_b0()
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, latent_size)
    elif model_type == 'xception':
        model = timm.create_model('xception')
        model.fc = torch.nn.Linear(model.fc.in_features, latent_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = FFTWrapper(base_model=model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    return model

def extract_features(model, image_paths, batch_size=32, img_size=256):
    transform = transforms.Compose([
        lambda img: transforms.Resize(img_size)(img) if min(img.size) < img_size else img,
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    features = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_imgs = [transform(Image.open(p).convert('RGB')) for p in batch_paths]
            batch_tensor = torch.stack(batch_imgs).to(device)
            batch_feats = model(img2noise(batch_tensor))
            features.append(batch_feats.cpu())
    return torch.cat(features, dim=0)

def compute_similarity_matrix(features1, features2, pair_classifier):

    with torch.no_grad():
        features1 = features1.to(device)
        features2 = features2.to(device)
        
        N1, N2 = features1.size(0), features2.size(0)
        
        feats1_exp = features1.unsqueeze(1).expand(-1, N2, -1)  # [N1, N2, D]
        feats2_exp = features2.unsqueeze(0).expand(N1, -1, -1)  # [N1, N2, D]
        
        pairs1 = feats1_exp.reshape(-1, features1.size(-1))  # [N1*N2, D]
        pairs2 = feats2_exp.reshape(-1, features2.size(-1))  # [N1*N2, D]
        
        logits = pair_classifier(pairs1, pairs2)
        probs = torch.sigmoid(logits).view(N1, N2)  # [N1, N2]
        
        return probs.cpu().numpy()

def prepare_datasets(dataset_dirs, num_images=100):
    S_datasets = []
    D_datasets = []
    
    for dataset_dir in dataset_dirs:
        image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if is_image_file(f)]
        
        if len(image_paths) < num_images:
            continue
        
        random.shuffle(image_paths)
        
        if len(image_paths) < 2 * num_images:
            D_images = image_paths[:num_images]
            S_images = image_paths[num_images:]
            print(f"{os.path.basename(dataset_dir)}: D={len(D_images)}, S={len(S_images)}")
        else:
            S_images = image_paths[:num_images]
            D_images = image_paths[num_images:2*num_images]
        
        S_datasets.append(S_images)
        D_datasets.append(D_images)
    
    return S_datasets, D_datasets

def compute_threshold_stats(similarity_matrix):
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    similarity_values = similarity_matrix[mask]
    
    mean_sim = np.mean(similarity_values)
    std_sim = np.std(similarity_values)
    
    return mean_sim, std_sim, similarity_values

def compute_accuracy(predictions, true_label, threshold):
    if true_label:
        correct = np.sum(predictions > threshold)
    else:
        correct = np.sum(predictions <= threshold)
    
    return correct / len(predictions)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{args.model_path}")
    feature_model = load_feature_model(args.model_path, args.model_type, args.latent_size).eval()
    
    pair_classifier = FeaturePairClassifier(feature_dim=args.latent_size).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'pair_classifier' in checkpoint:
        pair_classifier.load_state_dict(checkpoint['pair_classifier'])
    else:
        print("no pair_classifier")
    
    dataset_dirs = get_image_subdirs(args.dataset_dirs)
    
    print(f" {len(dataset_dirs)} datasets")
    
    name_to_id, id_to_name, id_to_path = create_class_mapping(dataset_dirs)
    
    for i, dataset_dir in enumerate(dataset_dirs):
        class_id = i + 1
        class_name = os.path.basename(dataset_dir)
        print(f"    {class_id}: {class_name}")
    
    mapping_data = []
    for i, dataset_dir in enumerate(dataset_dirs):
        class_id = i + 1
        class_name = os.path.basename(dataset_dir)
        mapping_data.append({
            'class_id': class_id,
            'class_name': class_name,
            'dataset_path': dataset_dir
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(os.path.join(args.output_dir, "class_mapping.csv"), index=False)
    
    S_datasets, D_datasets = prepare_datasets(dataset_dirs, args.num_images)
    num_classes = len(S_datasets)
    
    if num_classes == 0:
        return
    

    S_features = []
    for i, S_images in enumerate(S_datasets):
        class_id = i + 1
        class_name = id_to_name[class_id]
        print(f"  S_{class_id} ({class_name}: {len(S_images)})")
        features = extract_features(feature_model, S_images, args.batch_size, args.img_size)
        S_features.append(features)
    
    D_features = []
    for i, D_images in enumerate(D_datasets):
        class_id = i + 1
        class_name = id_to_name[class_id]
        print(f"  D_{class_id} ({class_name}: {len(D_images)})")
        features = extract_features(feature_model, D_images, args.batch_size, args.img_size)
        D_features.append(features)
    
    threshold_stats = []
    
    for i in range(num_classes):
        class_id = i + 1
        class_name = id_to_name[class_id]
        print(f"  SS_{class_id} ({class_name})")
        
        ss_matrix = compute_similarity_matrix(S_features[i], S_features[i], pair_classifier)
        
        ss_path = os.path.join(args.output_dir, f"SS_{class_id:03d}.csv")
        actual_size = len(S_features[i])
        ss_df = pd.DataFrame(ss_matrix, 
                        index=[f'S{class_id:03d}_img_{j:03d}' for j in range(actual_size)],
                        columns=[f'S{class_id:03d}_img_{j:03d}' for j in range(actual_size)])
        ss_df.to_csv(ss_path)
        
        mean_sim, std_sim, similarity_values = compute_threshold_stats(ss_matrix)
        threshold_stats.append({
            'class_id': class_id,
            'class_name': class_name,
            'mean': mean_sim,
            'std': std_sim,
            'threshold_k1': mean_sim - 1.0 * std_sim,
            'num_values': len(similarity_values)
        })
    
    threshold_df = pd.DataFrame(threshold_stats)
    threshold_df.to_csv(os.path.join(args.output_dir, "threshold_stats.csv"), index=False)
    
    all_predictions = []
    accuracy_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        class_i_id = i + 1
        class_i_name = id_to_name[class_i_id]
        print(f"  {class_i_id} ({class_i_name})")
        
        threshold = threshold_stats[i]['threshold_k1']
        
        class_predictions = []
        class_labels = []
        
        for j in range(num_classes):
            class_j_id = j + 1
            class_j_name = id_to_name[class_j_id]
            print(f"    SD_{class_i_id}_{class_j_id} ({class_i_name} vs {class_j_name})")
            
            # (S_i vs D_j)
            sd_matrix = compute_similarity_matrix(S_features[i], D_features[j], pair_classifier)
            
            sd_path = os.path.join(args.output_dir, f"SD_{class_i_id:03d}_{class_j_id:03d}.csv")
            s_size = len(S_features[i])  #
            d_size = len(D_features[j])  #
            sd_df = pd.DataFrame(sd_matrix,
                            index=[f'S{class_i_id:03d}_img_{k:03d}' for k in range(s_size)],
                            columns=[f'D{class_j_id:03d}_img_{k:03d}' for k in range(d_size)])
            sd_df.to_csv(sd_path)
            
            predictions = np.mean(sd_matrix, axis=0)  # [100]
            
            is_same_class = (i == j)
            accuracy = compute_accuracy(predictions, is_same_class, threshold)
            accuracy_matrix[i, j] = accuracy
            
            class_predictions.extend(predictions)
            class_labels.extend([1 if is_same_class else 0] * len(predictions))
        
        class_auc = roc_auc_score(class_labels, class_predictions)
        all_predictions.append({
            'class_id': class_i_id,
            'class_name': class_i_name,
            'auc': class_auc,
            'predictions': class_predictions,
            'labels': class_labels
        })
    
    accuracy_df = pd.DataFrame(accuracy_matrix,
                              index=[f'Class_{i+1:03d}_{id_to_name[i+1]}' for i in range(num_classes)],
                              columns=[f'D_{j+1:03d}_{id_to_name[j+1]}' for j in range(num_classes)])
    accuracy_df.to_csv(os.path.join(args.output_dir, "accuracy_matrix.csv"))
    
    auc_results = []
    for pred_data in all_predictions:
        auc_results.append({
            'class_id': pred_data['class_id'],
            'class_name': pred_data['class_name'],
            'auc': pred_data['auc']
        })
    
    auc_df = pd.DataFrame(auc_results)
    auc_df.to_csv(os.path.join(args.output_dir, "auc_results.csv"), index=False)
    
    mean_auc = np.mean([r['auc'] for r in auc_results])
    mean_accuracy = np.mean(np.diag(accuracy_matrix))

    
    with open(os.path.join(args.output_dir, "summary_stats.txt"), 'w') as f:
        f.write(f"mean_auc: {mean_auc}\n")
        f.write(f"mean_accuracy: {mean_accuracy}\n")
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"num_images_per_class: {args.num_images}\n")
        f.write(f"threshold_k: 1.0\n")
        f.write(f"class_mapping:\n")
        for class_id, class_name in id_to_name.items():
            f.write(f"  {class_id}: {class_name}\n")
    
    print(f"   AUC: {mean_auc:.4f}")
    print(f"   ACC: {mean_accuracy:.4f}")
    print(f"   NUM: {num_classes}")

    for result in auc_results:
        print(f"   {result['class_id']} ({result['class_name']}): {result['auc']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='step-0001000.pth')
    parser.add_argument('--model_type', type=str, default='res50', choices=['res50', 'effb0', 'xception'])
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='multi_class_eval-1k-aigc-nocrop')
    parser.add_argument('--dataset_dirs', type=str, default='AIGCDetectionBenchMark/test')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)