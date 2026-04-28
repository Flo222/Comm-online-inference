import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.geometry import warp_perspective
import matplotlib.pyplot as plt

from src.models.resnet import resnet18
from src.models.shufflenetv2 import shufflenet_v2_x0_5
from src.models.multiview_base import MultiviewBase
from src.utils.image_utils import img_color_denormalize, array2heatmap
from src.utils.projection import get_worldcoord_from_imgcoord_mat


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(
            nn.Conv2d(in_dim, feat_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, out_dim, 1)
        )
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc


class MVDet(MultiviewBase):
    def __init__(self, dataset, arch='resnet18', aggregation='max',
                 use_bottleneck=True, hidden_dim=128, outfeat_dim=0, z=0):
        super().__init__(dataset, aggregation)

        self.Rimg_shape = np.array(dataset.Rimg_shape)
        self.Rworld_shape = np.array(dataset.Rworld_shape)
        self.img_reduce = dataset.img_reduce

        world_zoom_mat = np.diag([dataset.world_reduce, dataset.world_reduce, 1])
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(
            dataset.base.worldcoord_from_worldgrid_mat
            @ world_zoom_mat
            @ dataset.base.world_indexing_from_xy_mat
        )

        worldcoord_from_imgcoord_mats = [
            get_worldcoord_from_imgcoord_mat(
                dataset.base.intrinsic_matrices[cam],
                dataset.base.extrinsic_matrices[cam],
                z / dataset.base.worldcoord_unit
            )
            for cam in range(dataset.num_cam)
        ]

        proj_mats = torch.stack([
            torch.from_numpy(Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[cam])
            for cam in range(dataset.num_cam)
        ]).float()
        self.register_buffer('proj_mats', proj_mats)

        if arch == 'resnet18':
            self.base = nn.Sequential(*list(
                resnet18(
                    pretrained=True,
                    replace_stride_with_dilation=[False, True, True]
                ).children()
            )[:-2])
            base_dim = 512
        elif arch == 'shufflenet0.5':
            self.base = nn.Sequential(*list(
                shufflenet_v2_x0_5(
                    pretrained=True,
                    replace_stride_with_dilation=[False, True, True]
                ).children()
            )[:-2])
            base_dim = 192
        else:
            raise ValueError('architecture currently supports [shufflenet0.5, resnet18]')

        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(base_dim, hidden_dim, 1),
                nn.ReLU()
            )
            base_dim = hidden_dim
        else:
            self.bottleneck = nn.Identity()

        # image heads
        self.img_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.img_offset = output_head(base_dim, outfeat_dim, 2)
        self.img_wh = output_head(base_dim, outfeat_dim, 2)

        # world feature tower
        self.world_feat = nn.Sequential(
            nn.Conv2d(base_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4), nn.ReLU(),
        )

        # world heads
        self.world_heatmap = output_head(hidden_dim, outfeat_dim, 1)
        self.world_offset = output_head(hidden_dim, outfeat_dim, 2)

        # init
        self.img_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.img_offset)
        fill_fc_weights(self.img_wh)
        self.world_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.world_offset)

    def get_feat(self, imgs, M, down=1, visualize=False, cam_indices=None):
        """
        imgs: [B, N, C, H, W]
        M:    [B, N, 3, 3]
        cam_indices:
            None -> 默认使用 [0, 1, ..., N-1]，兼容原始集中式写法
            list[int] -> 显式指定每一路输入图像对应的真实相机编号
        """
        B, N, _, _, _ = imgs.shape

        if cam_indices is None:
            cam_indices = list(range(N))
        if isinstance(cam_indices, int):
            cam_indices = [cam_indices]
        assert len(cam_indices) == N, \
            f"len(cam_indices)={len(cam_indices)} must equal N={N}"

        imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1 / down)

        inverse_affine_mats = torch.inverse(M.view(B * N, 3, 3))

        scale_mat = torch.diag(
            torch.tensor(
                [self.img_reduce * down, self.img_reduce * down, 1.0],
                dtype=inverse_affine_mats.dtype,
                device=inverse_affine_mats.device,
            )
        ).unsqueeze(0).repeat(B * N, 1, 1)

        imgcoord_from_Rimggrid_mat = inverse_affine_mats @ scale_mat

        cam_indices = torch.as_tensor(
            cam_indices,
            dtype=torch.long,
            device=self.proj_mats.device,
        )

        selected_proj_mats = self.proj_mats.index_select(0, cam_indices)

        proj_mats = (
            selected_proj_mats
            .to(device=imgs.device, dtype=imgs.dtype)
            .unsqueeze(0)
            .repeat(B, 1, 1, 1)
            .flatten(0, 1)
        ) @ imgcoord_from_Rimggrid_mat

        imgs_feat = self.base(imgs)
        imgs_feat = self.bottleneck(imgs_feat)

        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset = self.img_offset(imgs_feat)
        imgs_wh = self.img_wh(imgs_feat)

        world_feat = warp_perspective(
            imgs_feat,
            proj_mats,
            tuple(self.Rworld_shape.tolist()),
        ).unflatten(0, [B, N])

        aux_res = (
            F.interpolate(imgs_heatmap, tuple(self.Rimg_shape.tolist())),
            F.interpolate(imgs_offset, tuple(self.Rimg_shape.tolist())),
            F.interpolate(imgs_wh, tuple(self.Rimg_shape.tolist())),
        )

        return world_feat, aux_res


    def get_feat_single_cam_correct(self, img, affine_mat, cam_idx, down=1, visualize=False):
        """
        单节点真实推理接口。

        img:
            [C, H, W] or [B, C, H, W]
        affine_mat:
            [3, 3] or [B, 3, 3]
        cam_idx:
            当前节点对应的真实相机编号
        return:
            local_world_feat: [B, C, H_world, W_world]
            aux_res: image-level heatmap / offset / wh
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)          # [1, C, H, W]
        if affine_mat.dim() == 2:
            affine_mat = affine_mat.unsqueeze(0)  # [1, 3, 3]

        imgs = img.unsqueeze(1)             # [B, 1, C, H, W]
        affine_mats = affine_mat.unsqueeze(1)  # [B, 1, 3, 3]

        world_feat, aux_res = self.get_feat(
            imgs,
            affine_mats,
            down=down,
            visualize=visualize,
            cam_indices=[int(cam_idx)],
        )

        local_world_feat = world_feat[:, 0]  # [B, C, H_world, W_world]
        return local_world_feat, aux_res

    def get_output(self, world_feat, visualize=False):
        world_feat = self.world_feat(world_feat)
        world_heatmap = self.world_heatmap(world_feat)
        world_offset = self.world_offset(world_feat)

        if visualize:
            visualize_img = array2heatmap(torch.norm(world_feat[0].detach(), dim=0).cpu())
            plt.imshow(visualize_img)
            plt.show()
            visualize_img = array2heatmap(torch.sigmoid(world_heatmap.detach())[0, 0].cpu())
            plt.imshow(visualize_img)
            plt.show()

        return world_heatmap, world_offset