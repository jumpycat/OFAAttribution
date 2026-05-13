import math
import torch.nn as nn
import torch
import numpy as np
from PIL import Image


def conv_with_padding(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding = None):
    if padding is None:
        padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, dilation=dilation, padding=padding, bias=bias)

def conv_init(conv, act='linear'):
    r"""
    Reproduces conv initialization from DnCNN
    """
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n))

def batchnorm_init(m, kernelsize=3):
    r"""
    Reproduces batchnorm initialization from DnCNN
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()

def make_activation(act):
    if act is None:
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'linear':
        return None
    else:
        assert(False)

def make_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum = 0.1, padding=None):
    r"""
    :param nplanes_in: number of of input feature channels
    :param kernels: list of kernel size for convolution layers
    :param features: list of hidden layer feature channels
    :param bns: list of whether to add batchnorm layers
    :param acts: list of activations
    :param dilats: list of dilation factors
    :param bn_momentum: momentum of batchnorm
    :param padding: integer for padding (None for same padding)
    """

    depth = len(features)
    assert(len(features)==len(kernels))

    layers = list()
    for i in range(0,depth):
        if i==0:
            in_feats = nplanes_in
        else:
            in_feats = features[i-1]

        elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding, bias=not(bns[i]))
        conv_init(elem, act=acts[i])
        layers.append(elem)

        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum = bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)

        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)



class DnCNN(nn.Module):
    r"""
    Implements a DnCNN network
    """
    def __init__(self, nplanes_in, nplanes_out, features, kernel, depth, activation, residual, bn, lastact=None, bn_momentum = 0.10, padding=None):
        r"""
        :param nplanes_in: number of of input feature channels
        :param nplanes_out: number of of output feature channels
        :param features: number of of hidden layer feature channels
        :param kernel: kernel size of convolution layers
        :param depth: number of convolution layers (minimum 2)
        :param bn:  whether to add batchnorm layers
        :param residual: whether to add a residual connection from input to output
        :param bn_momentum: momentum of batchnorm
        :param padding: inteteger for padding
        """
        super(DnCNN, self).__init__()

        self.residual = residual
        self.nplanes_out = nplanes_out
        self.nplanes_in = nplanes_in

        kernels = [kernel, ] * depth
        features = [features, ] * (depth-1) + [nplanes_out, ]
        bns = [False, ] + [bn,] * (depth - 2) + [False, ]
        dilats = [1, ] * depth
        acts = [activation, ] * (depth - 1) + [lastact, ]
        self.layers = make_net(nplanes_in, kernels, features, bns, acts, dilats=dilats, bn_momentum = bn_momentum, padding=padding)


    def forward(self, x):
        shortcut = x

        x = self.layers(x)

        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out)
            x[:, :nshortcut, :, :] = x[:, :nshortcut, :, :] + shortcut[:, :nshortcut, :, :]

        return x


def add_commandline_networkparams(parser, name, features, depth, kernel, activation, bn):
    parser.add_argument("--{}.{}".format(name, "features"  ), type=int, default=features  )
    parser.add_argument("--{}.{}".format(name, "depth"     ), type=int, default=depth     )
    parser.add_argument("--{}.{}".format(name, "kernel"    ), type=int, default=kernel    )
    parser.add_argument("--{}.{}".format(name, "activation"), type=str, default=activation)

    bnarg = "{}.{}".format(name, "bn")
    parser.add_argument("--"+bnarg   , action="store_true", dest=bnarg)
    parser.add_argument("--{}.{}".format(name, "no-bn"), action="store_false", dest=bnarg)
    parser.set_defaults(**{bnarg: bn})




class get_denoiser:
    def __init__(self, sigma, cuda=False):
        num_levels = 17
        out_channel = 3
        network = make_net(3, kernels=[3, ] * num_levels,
                           features=[64, ] * (num_levels - 1) + [out_channel],
                           bns=[False, ] + [True, ] *
                           (num_levels - 2) + [False, ],
                           acts=['relu', ] * (num_levels - 1) + ['linear', ],
                           dilats=[1, ] * num_levels,
                           bn_momentum=0.1, padding=0)
        self.sigma = sigma
        if sigma == 1:
            weights_path = "model_best.th"
        else:
            print("Sigma should be one")
            assert False
        state_dict = torch.load(weights_path, torch.device('cpu'))
        network.load_state_dict(state_dict["network"])
        self.device = cuda if cuda else 'cpu'
        self.network = network.to(self.device).eval()

    def __call__(self, img):
        import torch
        with torch.no_grad():
            img = torch.from_numpy(np.float32(img)).permute(2, 0, 1)[None, ...]
            res = self.network(img.to(self.device))[
                0].permute(1, 2, 0).cpu().numpy()
        return res

    def denoise(self, img):
        import torch
        with torch.no_grad():
            img = torch.from_numpy(np.float32(img)).permute(2, 0, 1)[None, ...]
            img = img[:, :, 17:-17, 17:-17] - self.sigma / \
                256.0 * self.network(img.to(self.device)).cpu()
            img = img[0].permute(1, 2, 0).numpy()
        return img


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def inference_denoise(image_path, sigma=1, use_cuda=False):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0

    denoiser = get_denoiser(sigma=sigma, cuda=use_cuda)
    denoised_residual = denoiser(img_np)  
    print(denoised_residual.shape)

    h, w = denoised_residual.shape[:2]
    img_cropped = img_np[:h, :w, :]
    denoised_img = img_cropped - (sigma / 256.0) * denoised_residual

    denoised_img = np.clip(denoised_img, 0, 1)
    denoised_img_pil = Image.fromarray((denoised_img * 255).astype(np.uint8))

    denoised_residual = np.clip(denoised_residual, 0, 1)
    denoised_residual = Image.fromarray((denoised_residual * 255).astype(np.uint8))

    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Noise")
    plt.imshow(denoised_residual)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Denoised Image")
    plt.imshow(denoised_img_pil)
    plt.axis('off')

    plt.savefig("comparison.png", bbox_inches='tight')
    plt.close()

    return denoised_img_pil

if __name__ == "__main__":
    image_path = "000000008081.jpg"
    denoised_img = inference_denoise(image_path, sigma=1, use_cuda=False)
    denoised_img.save("denoised_output.jpg")
