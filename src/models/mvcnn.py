import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.multiview_base import MultiviewBase


class MVCNN(MultiviewBase):
    def __init__(self, dataset, arch='resnet18', aggregation='max'):
        super().__init__(dataset, aggregation)

        if arch == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            self.base = nn.Sequential(*list(backbone.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, dataset.num_class)
        elif arch == 'vgg11':
            backbone = models.vgg11(pretrained=True)
            self.base = backbone.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = backbone.classifier
            self.classifier[-1] = nn.Linear(4096, dataset.num_class)
        else:
            raise ValueError('architecture currently supports [vgg11, resnet18]')

    def get_feat(self, imgs, M=None, down=1, visualize=False):
        B, N, _, _, _ = imgs.shape
        imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1 / down)
        imgs_feat = self.base(imgs)
        imgs_feat = self.avgpool(imgs_feat)
        imgs_feat = imgs_feat.unflatten(0, [B, N])
        return imgs_feat, None

    def get_output(self, overall_feat, visualize=False):
        overall_feat = torch.flatten(overall_feat, 1)
        overall_result = self.classifier(overall_feat)
        return overall_result