import torch
import torch.nn as nn
import torchvision
import pytorch_ssim
from torch.nn import functional as F
from torchvision import models, transforms

from utils import SingletonPattern
from model import LayerOutputModelDecorator

from typing import List


class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.
    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.
    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).
    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.
    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.
    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        # self.model = self.models[model](pretrained=True).features[:layer+1]
        if model == 'vgg16':
            self.model = self.models[model](weights=models.VGG16_Weights.DEFAULT).features[:layer+1]
        else:
            self.model = self.models[model](weights=models.VGG19_Weights.DEFAULT).features[:layer+1]

        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)
    

VGG_Loss = VGGLoss(layer=15).cuda()

'''
This file is directly imported from NSRR-PyTorch(https://github.com/IMAC-projects/NSRR-PyTorch)
'''
def feature_reconstruction_loss(conv_layer_output: torch.Tensor, conv_layer_target: torch.Tensor) -> torch.Tensor:
    """
    Computes Feature Reconstruction Loss as defined in Johnson et al. (2016)
    todo: syntax
    Justin Johnson, Alexandre Alahi, and Li Fei-Fei. 2016. Perceptual losses for real-time
    style transfer and super-resolution. In European Conference on Computer Vision.
    694–711.
    Takes the already-computed output from the VGG16 convolution layers.
    """
    if conv_layer_output.shape != conv_layer_target.shape:
        raise ValueError("Output and target tensors have different dimensions!"+str(conv_layer_output.shape)+","+str(conv_layer_target.shape))
    loss = conv_layer_output.dist(conv_layer_target, p=2) / torch.numel(conv_layer_output)
    return loss


def nsrr_loss(output: torch.Tensor, target: torch.Tensor, w: float=1e-3) -> torch.Tensor:
    """
    Computes the loss as defined in the NSRR paper.
    
    """
    loss_ssim = 1 - pytorch_ssim.ssim(output, target)
    loss_perception = 0
    # conv_layers_output = PerceptualLossManager().get_vgg16_conv_layers_output(output)
    # conv_layers_target = PerceptualLossManager().get_vgg16_conv_layers_output(target)
    # for i in range(len(conv_layers_output)):
    #     loss_perception += feature_reconstruction_loss(conv_layers_output[i], conv_layers_target[i])
    loss_perception = VGG_Loss(output, target)

    loss_l1 = torch.abs(output - target).mean()

    loss = 0.4 * loss_l1 + loss_ssim + w * loss_perception
    return loss

class PerceptualLossManager(metaclass=SingletonPattern):
    """
    Singleton
    """
    # Init
    def __init__(self):
        self.vgg_model = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg_model.eval()
        """ 
            Feature Reconstruction Loss 
            - needs output from each convolution layer.
        """
        self.layer_predicate = lambda name, module: type(module) == nn.Conv2d
        self.lom = LayerOutputModelDecorator(self.vgg_model.features, self.layer_predicate)

    def get_vgg16_conv_layers_output(self, x: torch.Tensor)-> List[torch.Tensor]:
        """
        Returns the list of output of x on the pre-trained VGG16 model for each convolution layer.
        """
        return self.lom.forward(x)

