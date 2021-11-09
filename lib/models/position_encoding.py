# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

# from lib.models.util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of
    the position embedding, very similar to the one
    used by the Attention is all you need paper,
    generalized to work on images.
    """
    def __init__(self, num_pos_feats=64,
                 temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = x.new_zeros((x.size()[0], ) + x.size()[2:]).bool()
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# class Embedding_sincosin(nn.Module):
#     def __init__(self, in_channels, N_freqs, log_sampling=True):
#         """
#         Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
#         in_channels: number of input channels (3 for both xyz and direction)
#         """
#         super(Embedding_jianfeng, self).__init__()
#         self.N_freqs = N_freqs
#         self.in_channels = in_channels
#         self.funcs = [torch.sin, torch.cos]
#         self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
#
#         if log_sampling:
#             self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
#         else:
#             self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
#
#     def forward(self, x):
#         """
#         Embeds x to (x, sin(2^k x), cos(2^k x), ...)
#         Inputs:
#             x: (B, self.in_channels)
#         Outputs:
#             out: (B, self.out_channels)
#         """
#         out = [x]
#         for freq in self.freq_bands:
#             for func in self.funcs:
#                 out += [func(freq * x)]
#
#         return torch.cat(out, -1)


class PositionEmbeddingSine_Ray(nn.Module):
    """
    This is a more standard version of the
    position embedding, very similar to the one
    used by the Attention is all you need paper,
    generalized to work on images.
    """
    def __init__(self, num_pos_feats=64,
                 temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        x_embed = x[..., 0:1] * self.scale
        y_embed = x[..., 1:2] * self.scale
        z_embed = x[..., 2:3] * self.scale

        dim_t_x = torch.arange(80, dtype=torch.float32, device=x.device)
        dim_t_x = self.temperature ** (2 * (dim_t_x // 2) / 80)
        dim_t_y = torch.arange(80, dtype=torch.float32, device=x.device)
        dim_t_y = self.temperature ** (2 * (dim_t_y // 2) / 80)
        dim_t_z = torch.arange(96, dtype=torch.float32, device=x.device)
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / 96)

        pos_x = x_embed / dim_t_x
        pos_y = y_embed / dim_t_y
        pos_z = z_embed / dim_t_z

        pos_x = torch.stack((pos_x[..., 0::2].sin(),
                             pos_x[..., 1::2].cos()), dim=3).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(),
                             pos_y[..., 1::2].cos()), dim=3).flatten(3)
        pos_z = torch.stack((pos_z[..., 0::2].sin(),
                             pos_z[..., 1::2].cos()), dim=3).flatten(3)

        pos = torch.cat((pos_y, pos_x, pos_z), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """

    j, i = torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W))
    # the direction here is without +0.5 pixel centering
    # as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    # directions = torch.stack([(i.to(focal.device)-W*.5)/focal,
    # -(j.to(focal.device)-H*.5)/focal, -torch.ones_like(i)], -1)  # (H, W, 3)
    directions = torch.stack([(i.to(focal.device) - W * .5),
                              -(j.to(focal.device) - H * .5),
                              -torch.ones_like(i).to(focal.device)], -1)
    directions \
        = directions.unsqueeze(0).unsqueeze(0) / focal.unsqueeze(2).unsqueeze(3)
    return directions


def get_rays(H, W, focal, cam_R, cam_T):
    """
    Get ray origin and normalized directions in world
    coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from
        camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of
        the rays in world coordinate
    """
    directions = get_ray_directions(H, W, focal.float())
    org_shape = directions.shape
    rays_d = (directions.flatten(2, 3) @ cam_R.float()).view(org_shape)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_d


def get_2d_coords(image_size, H, W, K, R, T, ret_rays_o=False):
    # calculate the camera origin
    ratio = W / image_size[0]
    # batch = K.size(0)
    views = K.size(1)
    K = K.reshape(-1, 3, 3).float()
    R = R.reshape(-1, 3, 3).float()
    T = T.reshape(-1, 3, 1).float()
    # re-scale camera parameters
    K[:, :2] *= ratio
    # rays_o = -torch.bmm(R.transpose(2, 1), T)
    # calculate the world coordinates of pixels
    j, i = torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W))
    xy = torch.stack([i.to(K.device)/W, j.to(K.device)/H], dim=-1).unsqueeze(0)
    return xy.unsqueeze(0).expand(-1, views, -1, -1, -1)


def get_rays_new(image_size, H, W, K, R, T, ret_rays_o=False):
    # calculate the camera origin
    ratio = W / image_size[0]
    batch = K.size(0)
    views = K.size(1)
    K = K.reshape(-1, 3, 3).float()
    R = R.reshape(-1, 3, 3).float()
    T = T.reshape(-1, 3, 1).float()
    # re-scale camera parameters
    K[:, :2] *= ratio
    rays_o = -torch.bmm(R.transpose(2, 1), T)
    # calculate the world coordinates of pixels
    j, i = torch.meshgrid(torch.linspace(0, H-1, H),
                          torch.linspace(0, W-1, W))
    xy1 = torch.stack([i.to(K.device), j.to(K.device),
                       torch.ones_like(i).to(K.device)], dim=-1).unsqueeze(0)
    pixel_camera = torch.bmm(xy1.flatten(1, 2).repeat(views, 1, 1),
                             torch.inverse(K).transpose(2, 1))
    pixel_world = torch.bmm(pixel_camera-T.transpose(2, 1), R)
    rays_d = pixel_world - rays_o.transpose(2, 1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = rays_o.unsqueeze(1).repeat(1, H*W, 1, 1)
    if ret_rays_o:
        return rays_d.reshape(batch, views, H, W, 3), \
               rays_o.reshape(batch, views, H, W, 3) / 1000
    else:
        return rays_d.reshape(batch, views, H, W, 3)

    """ Numpy Code
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d
    """
