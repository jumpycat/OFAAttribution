import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime, timedelta
import time
import importlib
import random
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torchvision import models
import timm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'

def supervised_contrastive_loss(features, labels, temperature=0.5):
    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(mask.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(similarity) * logits_mask  
    log_prob = similarity - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    loss = -mean_log_prob_pos.mean()
    return loss

    
def supervised_contrastive_loss_with_bank(features, labels, memory_bank=None, temperature=0.5):
    features = F.normalize(features, dim=1)
    batch_size = features.size(0)
    
    similarity = torch.matmul(features, features.T) / temperature
    
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(mask.device)
    mask = mask * logits_mask
    
    if memory_bank is not None:
        bank_features = memory_bank.get_features()
        if bank_features.size(0) > 0:
            bank_features = F.normalize(bank_features, dim=1)
            bank_similarity = torch.matmul(features, bank_features.T) / temperature
            similarity = torch.cat([similarity, bank_similarity], dim=1)
    
    exp_sim = torch.exp(similarity)
    if memory_bank is not None:
        exp_sim_batch = exp_sim[:, :batch_size] * logits_mask
        exp_sim_bank = exp_sim[:, batch_size:] if exp_sim.size(1) > batch_size else torch.zeros_like(exp_sim[:, :0])
        total_exp_sim = exp_sim_batch.sum(1, keepdim=True) + exp_sim_bank.sum(1, keepdim=True)
    else:
        total_exp_sim = (exp_sim * logits_mask).sum(1, keepdim=True)
    
    log_prob = similarity[:, :batch_size] - torch.log(total_exp_sim + 1e-8)
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    
    return loss

class FeaturePairClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        logits = self.fc(x)
        return logits.squeeze(1)

class Classifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.fc(x)
        return logits.squeeze(1)


def generate_balanced_pairs(features, labels, neg_ratio=1):
    pairs_f1 = []
    pairs_f2 = []
    pair_labels = []

    batch_size = features.size(0)
    pos_pairs = []
    neg_pairs = []

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            label = 1.0 if labels[i] == labels[j] else 0.0
            if label == 1.0:
                pos_pairs.append((features[i], features[j], label))
            else:
                neg_pairs.append((features[i], features[j], label))

    num_pos = len(pos_pairs)
    num_neg = min(len(neg_pairs), num_pos * neg_ratio)
    neg_samples = random.sample(neg_pairs, num_neg)

    selected_pairs = pos_pairs + neg_samples
    random.shuffle(selected_pairs)

    for f1, f2, lbl in selected_pairs:
        pairs_f1.append(f1)
        pairs_f2.append(f2)
        pair_labels.append(lbl)

    pairs_f1 = torch.stack(pairs_f1)
    pairs_f2 = torch.stack(pairs_f2)
    pair_labels = torch.tensor(pair_labels, dtype=torch.float32).to(features.device)

    return pairs_f1, pairs_f2, pair_labels


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
        x = img2noise(x)
        x = get_log_magnitude(x)
        return self.base(x)


class MemoryBank:
    def __init__(self, size=4096, feature_dim=128, device=device):
        self.size = size
        self.feature_dim = feature_dim
        self.device = device
        
        self.features = torch.randn(size, feature_dim).to(device)
        self.features = F.normalize(self.features, dim=1)
        
        self.ptr = 0
        self.is_full = False
        
    def update(self, features):
        batch_size = features.size(0)  
        
        if batch_size >= self.size:
            self.features = features[-self.size:].clone()
            self.ptr = 0
            self.is_full = True
        else:
            if self.ptr + batch_size <= self.size: 
                self.features[self.ptr:self.ptr + batch_size] = features.clone()
                self.ptr += batch_size
            else:
                remaining = self.size - self.ptr 
                self.features[self.ptr:] = features[:remaining].clone()
                self.features[:batch_size - remaining] = features[remaining:].clone()
                self.ptr = batch_size - remaining
                
            if self.ptr == self.size:
                self.ptr = 0
                self.is_full = True
    
    def get_features(self):
        if self.is_full:
            return self.features
        else:
            if self.ptr > 0:
                return self.features[:self.ptr]
            else:
                return torch.empty(0, self.feature_dim).to(self.device)
            
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='CoCo/coco_train/train_2017')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--label_size', type=int, default=8)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--temp', type=float, default=0.02)
    parser.add_argument('--memory_bank_size', type=int, default=65536, help='Memory bank size')

    parser.add_argument('--diff_lambda', type=float, default=1.)
    parser.add_argument('--same_lambda', type=float, default=1.)
    parser.add_argument('--pair_lambda', type=float, default=1.)
    parser.add_argument('--cls_lambda', type=float, default=1.)


    parser.add_argument('--if_norm', type=float, default=1.)

    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--model_type', type=str, default='res50', choices=['res50','effb0','xception'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adamw', 'adam', 'sgd'])  
    parser.add_argument('--loss_type', type=str, default='sup', choices=['clip', 'sup', 'sgd'])


    parser.add_argument("--encoder_module", type=str, default="encoder2")
    parser.add_argument("--decoder_module", type=str, default="decoder24_with_graph")
    parser.add_argument('--noiser_path', type=str, default='nosier_250708204953_decoder24_bs4_ga32_both_1.0_0.25_lr4e-05/models/checkpoint-1000000.pth')

    return parser

def main(params):
    now = datetime.now().strftime('%y%m%d%H%M%S')
    exp_path = os.path.join('runs', f'extracter_{now}_contrav3_cls_{params.model_type}_bs{params.batch_size}_lb{params.label_size}_lt{params.latent_size}_tmp{params.temp}_lr{params.lr}')
    model_dir = os.path.join(exp_path, 'models')
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(exp_path, 'args.txt'), 'w') as f:
        for arg in vars(params):
            f.write(f'{arg}: {getattr(params, arg)}\n')
        f.write(f'Command: {" ".join(sys.argv)}\n')


    normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
    transform = transforms.Compose([
        lambda img: transforms.Resize(params.img_size)(img) if min(img.size) < params.img_size else img,
        transforms.RandomCrop(params.img_size),
        transforms.ToTensor(),
        normalize_vqgan
    ])
    dataset = ImageFolder(params.train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=16, drop_last=True, persistent_workers=True)

    if params.model_type == 'res50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, params.latent_size)
        model = model.to(device)
        model.train()
        
    elif params.model_type == 'effb0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, params.latent_size)
        model = model.to(device)
        model.train()

    elif params.model_type == 'xception':
        model = timm.create_model('xception', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, params.latent_size)
        model = model.to(device)
        model.train()

    model = FFTWrapper(base_model=model).train()


    pair_classifier = FeaturePairClassifier(feature_dim=params.latent_size).to(device).train()
    cls_classifier = Classifier(feature_dim=params.latent_size).to(device).train()

    bce_loss = nn.BCEWithLogitsLoss()


    param_opt = list(cls_classifier.parameters()) + list(model.parameters()) + list(pair_classifier.parameters())
    if params.optimizer == 'adam':
        optimizer = optim.Adam(param_opt, lr=params.lr)
    elif params.optimizer == 'sgd':
        optimizer = optim.SGD(param_opt, lr=params.lr)
    elif params.optimizer == 'adamw':
        optimizer = optim.AdamW(param_opt, lr=params.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {params.optimizer}")


    if params.resume is not None and os.path.isfile(params.resume):
        checkpoint = torch.load(params.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        try:
            pair_classifier.load_state_dict(checkpoint['pair_classifier'])
        except:
            print('pair_classifier')


    if params.noiser_path is not None and os.path.isfile(params.noiser_path):
        decoder_module = importlib.import_module(params.decoder_module)
        Decoder_class = getattr(decoder_module, "Decoder")

        encoder_module = importlib.import_module(params.encoder_module)
        Encoder_class = getattr(encoder_module, "Encoder")

        encoder = Encoder_class().to(device).eval()
        decoder = Decoder_class().to(device).eval()

        checkpoint = torch.load(params.noiser_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])


    start_time = time.time()
    data_iter = iter(loader)

    if params.if_norm == 1:
        model.train()
        print('train mode')
    else:
        model.eval()
        print('eval mode')

    memory_bank = MemoryBank(params.memory_bank_size, params.latent_size, device)

    for step in range(1, params.steps + 1):
        ########################################################
        all_imgs = []
        all_real = []
        all_labels = []

        for i in range(params.label_size):
            try:
                imgs, _ = next(data_iter)
            except:
                data_iter = iter(loader)
                imgs, _ = next(data_iter)
            imgs = imgs.to(device)

            with torch.no_grad():
                latents = encoder(imgs)
                recon, _ = decoder(latents).detach()  # [B, C, H, W]

            all_imgs.append(recon)
            all_real.append(imgs)
            all_labels.append(torch.full((recon.size(0),), i, dtype=torch.long, device=device))

        try:
            imgs, _ = next(data_iter)
        except:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)
        imgs = imgs.to(device)

        all_imgs.append(imgs)
        all_labels.append(torch.full((imgs.size(0),), params.label_size, dtype=torch.long, device=device))

        ##############################################
        all_imgs_tensor = torch.cat(all_imgs, dim=0)  # [B_total, C, H, W]
        all_labels_tensor = torch.cat(all_labels, dim=0)  # [B_total]
        all_real = torch.cat(all_real, dim=0)  # [B_total, C, H, W]
        all_feats_tensor = model(all_imgs_tensor)  # [B_total, D]
        real_outputs = model(all_real)

        with torch.no_grad():
            memory_bank.update(all_feats_tensor.detach())
            

        real_gt = torch.zeros(all_real.size(0), dtype=torch.float32).to(device)
        fake_gt = torch.ones(all_feats_tensor.size(0)-imgs.size(0), dtype=torch.float32).to(device)
        real_out_same = cls_classifier(real_outputs).view(-1).float()
        fake_out_same = cls_classifier(all_feats_tensor[:-imgs.size(0)]).view(-1).float()

        cls_loss_same = bce_loss(real_out_same, real_gt) + bce_loss(fake_out_same, fake_gt)

        loss_diff = supervised_contrastive_loss_with_bank(all_feats_tensor, all_labels_tensor, memory_bank, params.temp)


        pairs_f1, pairs_f2, pair_labels1 = generate_balanced_pairs(all_feats_tensor, all_labels_tensor)
        logits1 = pair_classifier(pairs_f1, pairs_f2)
        loss_pair1 = bce_loss(logits1, pair_labels1)


        if step % params.log_freq == 0:    
            with torch.no_grad():
                # === SAME ===
                real_correct_same = ((real_out_same > 0).float() == real_gt).sum().item()
                fake_correct_same = ((fake_out_same > 0).float() == fake_gt).sum().item()
                acc_same = (real_correct_same + fake_correct_same) / (real_gt.size(0) + fake_gt.size(0))


        ########################################################
        loss_same = 0.0
        if params.same_lambda > 0:
            try:
                imgs, _ = next(data_iter)
            except:
                data_iter = iter(loader)
                imgs, _ = next(data_iter)
            imgs = imgs.to(device)

            with torch.no_grad():
                latents = encoder(imgs)


            all_recons = []
            all_labels = []

            for i in range(params.label_size):
                with torch.no_grad():
                    recon, _ = decoder(latents).detach()  # [B, C, H, W]

                    all_recons.append(recon)
                    all_labels.append(torch.full((recon.size(0),), i, dtype=torch.long, device=device))

            all_imgs_tensor = torch.cat(all_recons, dim=0)     # [B * num_decoders, C, H, W]
            all_labels_tensor = torch.cat(all_labels, dim=0)   # [B * num_decoders]
            all_feats_tensor = model(all_imgs_tensor)      # [B * num_decoders, D]
            real_outputs = model(imgs)

            real_gt = torch.zeros(real_outputs.size(0), dtype=torch.float32).to(device)
            fake_gt = torch.ones(real_outputs.size(0), dtype=torch.float32).to(device)
            real_out_diff = cls_classifier(real_outputs).view(-1).float()
            fake_out_diff = cls_classifier(all_feats_tensor[:real_outputs.size(0)]).view(-1).float()

            cls_loss_diff = bce_loss(real_out_diff, real_gt) + bce_loss(fake_out_diff, fake_gt)

            if step % params.log_freq == 0:    
                with torch.no_grad():
                    # === DIFF ===
                    real_correct_diff = ((real_out_diff > 0).float() == real_gt).sum().item()
                    fake_correct_diff = ((fake_out_diff > 0).float() == fake_gt).sum().item()
                    acc_diff = (real_correct_diff + fake_correct_diff) / (real_gt.size(0) + fake_gt.size(0))


        loss_same = supervised_contrastive_loss(all_feats_tensor, all_labels_tensor, params.temp)

        pairs_f1, pairs_f2, pair_labels2 = generate_balanced_pairs(all_feats_tensor, all_labels_tensor)
        logits2 = pair_classifier(pairs_f1, pairs_f2)
        loss_pair2 = bce_loss(logits2, pair_labels2)


        loss = params.diff_lambda * loss_diff + params.same_lambda * loss_same + params.pair_lambda * loss_pair1 + params.pair_lambda * loss_pair2 + params.cls_lambda * cls_loss_same + params.cls_lambda * cls_loss_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % params.log_freq == 0:    
            duration = time.time() - start_time

            with torch.no_grad():
                preds = (torch.sigmoid(logits1) > 0.5).float()
                correct = (preds == pair_labels1).sum().item()
                total = pair_labels1.size(0)
                acc1 = correct / total

                preds = (torch.sigmoid(logits2) > 0.5).float()
                correct = (preds == pair_labels2).sum().item()
                total = pair_labels2.size(0)
                acc2 = correct / total

            log_msg = (
                f"{now} Step {step:07d} | "
                f"diff: {loss_diff:.5f} | "
                f"same: {loss_same:.5f} | "
                f"pair1: {loss_pair1:.5f} | "
                f"acc1: {acc1*100:.2f} | "
                f"pair2: {loss_pair2:.5f} | "
                f"acc2: {acc2*100:.2f} | "
                f"cls1: {cls_loss_same:.5f} | "
                f"acc1: {acc_same*100:.2f} | "
                f"cls2: {cls_loss_diff:.5f} | "
                f"acc2: {acc_diff*100:.2f} | "
                f"Time: {format_time(duration)}"
            )

            print(log_msg)
            with open(os.path.join(exp_path, 'logs.txt'), 'a') as f:
                f.write(log_msg + "\n")

        if step % params.save_freq == 0:
            save_path = os.path.join(model_dir, f"step-{step:07d}.pth")
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'pair_classifier':pair_classifier.state_dict(),
                'cls_classifier': cls_classifier.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
