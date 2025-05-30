# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")
import torch.nn.functional as F
import cv2 
import os 
import numpy as np
t1=torch.rand(21).argsort()
t2=torch.rand(32).argsort()
rnd_pts=list(zip(t1,t2))
vis_rnd_pts=[(i.item(), j.item()) for (i,j) in rnd_pts]
num_vis=30


def vis_attn_map(attention_maps, img_target, img_source, count, save_path='./vis_ca_map'):
    
    ########################## VIS CROSS ATTN MAPS (START) ###############################
    
    b, _, H, W = img_target.shape 
    attn_maps = torch.stack(attention_maps, dim=1)  # b 12 196 196 (twelve layers of already head averaged attention maps)

    p_size=16
    pH=H//p_size  # num patch H
    pW=W//p_size  # num patch W     

    for batch in range(b):  
        img_t = img_target[batch] # 3 224 224 
        img_s = img_source[batch] 
        attn_map = attn_maps[batch] # 12 196 196

        attn_map = attn_map.mean(dim=0) # average all layers of attention maps 

        np_img_s = (img_s-img_s.min()) / (img_s.max()-img_s.min()) * 255.0 # [0,255]
        np_img_t = (img_t-img_t.min())/(img_t.max()-img_t.min()) * 255.0   # [0,255]
        np_img_s = np_img_s.squeeze().permute(1,2,0).detach().cpu().numpy() # 224 224 3 
        np_img_t = np_img_t.squeeze().permute(1,2,0).detach().cpu().numpy()

        # List to store all visualizations
        all_vis_imgs = []
        
        for points in vis_rnd_pts[:num_vis]:
            idx_h=points[0]     # to vis idx_h
            idx_w=points[1]     # to vis idx_w
            idx_n=idx_h*pW+idx_w  # to vis token idx
            
            # plot white pixel to vis tkn location
            vis_np_img_s = np_img_s.copy()  # same as clone()
            vis_np_img_s[idx_h*p_size:(idx_h+1)*p_size, idx_w*p_size:(idx_w+1)*p_size,:]=255    # color with white pixel
            
            # breakpoint()
            # generate attn heat map
            attn_msk=attn_map[idx_n]  # hw=14*14=196
            # attn_msk[0]=0
            # attn_msk=attn_msk.softmax(dim=-1)
            attn_msk=attn_msk.view(1,1,pH,pW)
            attn_msk=F.interpolate(attn_msk, size=(H,W), mode='bilinear', align_corners=True)   # 224 224
            attn_msk=(attn_msk-attn_msk.min())/(attn_msk.max()-attn_msk.min())  # [0,1]
            attn_msk=attn_msk.squeeze().detach().cpu().numpy()*255  # [0,255]
            heat_mask=cv2.applyColorMap(attn_msk.astype(np.uint8), cv2.COLORMAP_JET)
            
            # overlap heat_mask to source image
            img_t_attn_msked = np_img_t[...,::-1] + heat_mask
            img_t_attn_msked = (img_t_attn_msked-img_t_attn_msked.min())/(img_t_attn_msked.max()-img_t_attn_msked.min())*255.0
            
            # Concatenate source and target images horizontally for this point
            combined_img = np.concatenate([vis_np_img_s[:,:,[2,1,0]], img_t_attn_msked], axis=1)
            all_vis_imgs.append(combined_img)
        
        # Stack all visualizations vertically
        final_vis = np.concatenate(all_vis_imgs, axis=0)
        
        # Save the combined visualization
        log_img_path = save_path
        if not os.path.exists(log_img_path):
            os.makedirs(log_img_path)
        cv2.imwrite(f'{log_img_path}/count{count}_batch{batch}_all_points.jpg', final_vis)


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 temperature=3.0,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.count = 0
        self.reciprocity = True

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        self.temperature = temperature

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        
        x_croco, pos = self.patch_embed(image, true_shape=true_shape) 
        B, _, H, W = image.shape
        patch_size = self.patch_embed.patch_size[0]

        if self.is_student:
            image = self.input_transform(image)
            x = self.model.patch_embed(image)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization

        if self.is_student:
            for i, (blk, enc_blk) in enumerate(zip(self.blocks, self.enc_blocks)):
                x = blk(x)
                # x_croco = enc_blk(x_croco, pos)
        else:
            for i, enc_blk in enumerate(self.enc_blocks):
                x_croco = enc_blk(x_croco, pos)

            x_croco = self.enc_norm(x_croco)

        # intermediate_feats into stack tensor and mean

        if self.is_student:
            x = self.norm(x)
            x = x[:, 1:]

            _, _, C = x.shape

            x_reshape = x.reshape(B, H // patch_size, W // patch_size, -1).permute(0, 3, 1, 2)
            x_reshape = self.refine_conv(x_reshape)
            x = x_reshape.permute(0, 2, 3, 1).reshape(B, -1, C)

            x = self.proj_layer(x)

            if self.use_adaptive_norm:
                x = self.student_adaptive_norm(x)
                intermediate_feats = x

            # outputs = self.model._intermediate_layers(image, [0,1,2,3])
            # outputs = [self.model.norm(out) for out in outputs]
            # outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]
            # outputs = [self.proj_layer(out) for out in outputs]      
            # return x, pos, None
            return x_croco, pos, intermediate_feats
    
        # return x_croco, pos, None
        intermediate_feats = x_croco
        return x_croco, pos, intermediate_feats

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, in_feats = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
            in_feat1, in_feat2 = in_feats.chunk(2, dim=0)
        else:
            out, pos, in_feat1 = self._encode_image(img1, true_shape1)
            out2, pos2, in_feat2 = self._encode_image(img2, true_shape2)
        # return out, out2, pos, pos2
        return out, out2, pos, pos2, in_feat1, in_feat2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2, in_feat1, in_feat2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
            in_feat1, in_feat2 = interleave(in_feat1, in_feat2)
        else:
            feat1, feat2, pos1, pos2, in_feat1, in_feat2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2), (in_feat1, in_feat2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        camaps1 = []
        camaps2 = []

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _, camap1 = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _, camap2 = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))
            camaps1.append(camap1)
            camaps2.append(camap2)

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        # return zip(*final_output)
        return list(zip(*final_output)), camaps1, camaps2

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2), (in_feat1, in_feat2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        # dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        dec_feats, tgt_camap, src_camap = self._decoder(feat1, pos1, feat2, pos2)
        dec1, dec2 = dec_feats[0], dec_feats[1]

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res1['in_feats'] = feat1
        res2['in_feats'] = feat2

        if self.reciprocity:
            tgt_camap = [attn.mean(dim=1).detach() for attn in tgt_camap]
            src_camap = [attn.mean(dim=1).detach() for attn in src_camap]
            tgt_camap = [(camap_t + camap_s.transpose(-1,-2))/2 for (camap_t, camap_s) in zip(tgt_camap, src_camap)]
        
            # tgt_camap = [camap.softmax(dim=-1) for camap in tgt_camap]
            tgt_camap = [(camap / self.temperature).softmax(dim=-1) for camap in tgt_camap]
            for i in range(len(tgt_camap)):
                tgt_camap[i][:,:,0]= tgt_camap[i].min()
        else:
            tgt_camap = [attn.mean(dim=1).detach() for attn in tgt_camap]   
            # heuristic attention refine
            for i in range(len(tgt_camap)):
                tgt_camap[i][:,:,0]= tgt_camap[i].min()   # b 196 196

        self.count += 1
        # vis_attn_map(tgt_camap, view2['img'], view1['img'], self.count, save_path='./visualization/camap')
        tgt_attn_map = torch.stack(tgt_camap, dim=1).mean(dim=1)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        res2['tgt_attn_map'] = tgt_attn_map
        return res1, res2
