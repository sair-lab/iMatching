from collections import defaultdict
import pprint
from pathlib import Path

import torch
import numpy as np
from einops.einops import rearrange

from ext.aspanformer.src.ASpanFormer.aspanformer import ASpanFormer
from ext.aspanformer.src.ASpanFormer.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from ext.aspanformer.src.losses.aspan_loss import ASpanLoss
from ext.aspanformer.src.optimizers import build_optimizer, build_scheduler
from ext.aspanformer.src.utils.metrics import (
    compute_symmetrical_epipolar_errors,compute_symmetrical_epipolar_errors_offset_bidirectional,
    compute_pose_errors,
    aggregate_metrics
)
from ext.aspanformer.src.utils.plotting import make_matching_figures,make_matching_figures_offset
from ext.aspanformer.src.utils.comm import gather, all_gather
from ext.aspanformer.src.utils.misc import lower_config, flattenList
from ext.aspanformer.src.utils.profiler import PassThroughProfiler
from ext.aspanformer.src.config.default import get_cfg_defaults

from utils.config import hydra_config, hydra_instantiable
from utils.misc import dictseq2seqdict, pad_to_same

# soft coarse match
from ext.aspanformer.src.ASpanFormer.utils.coarse_matching import mask_border, mask_border_with_padding

def soft_coarse_match(coarse_module, conf_matrix, data):
    """
    Args:
        conf_matrix (torch.Tensor): [N, L, S]
        data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
    Returns:
        coarse_matches (dict): {
            'b_ids' (torch.Tensor): [M'],
            'i_ids' (torch.Tensor): [M'],
            'j_ids' (torch.Tensor): [M'],
            'gt_mask' (torch.Tensor): [M'],
            'm_bids' (torch.Tensor): [M],
            'mkpts0_c' (torch.Tensor): [M, 2],
            'mkpts1_c' (torch.Tensor): [M, 2],
            'mconf' (torch.Tensor): [M]}
    """
    axes_lengths = {
        'h0c': data['hw0_c'][0],
        'w0c': data['hw0_c'][1],
        'h1c': data['hw1_c'][0],
        'w1c': data['hw1_c'][1]
    }
    _device = conf_matrix.device
    # 1. confidence thresholding
    mask = conf_matrix > coarse_module.thr
    mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                        **axes_lengths)
    if 'mask0' not in data:
        mask_border(mask, coarse_module.border_rm, False)
    else:
        mask_border_with_padding(mask, coarse_module.border_rm, False,
                                    data['mask0'], data['mask1'])
    mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                        **axes_lengths)

    # 2. mutual nearest
    mask = mask \
        * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
        * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

    # 3. find all valid coarse matches
    # this only works when at most one `True` in each row
    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mconf = conf_matrix[b_ids, i_ids, j_ids]

    # zitong
    i_weights = data['softmax_1']
    j_weights = data['softmax_2']
    indices_i = torch.cartesian_prod(torch.arange(axes_lengths['h0c']), 
                                     torch.arange(axes_lengths['w0c'])).to(_device)
    indices_j = torch.cartesian_prod(torch.arange(axes_lengths['h1c']), 
                                     torch.arange(axes_lengths['w1c'])).to(_device)

    soft_index_i = (indices_i[None, :, None] * i_weights[..., None]).sum(-3)[b_ids, j_ids]
    soft_index_j = (indices_j[None, None, :] * j_weights[..., None]).sum(-2)[b_ids, i_ids]

    scale = data['hw0_i'][0] / data['hw0_c'][0]
    scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
    soft_kpts0_c = soft_index_i * scale0
    soft_kpts1_c = soft_index_j * scale1

    # These matches is the current prediction (for visualization)
    coarse_matches = {
        'coord1_ec': soft_kpts0_c[..., [1, 0]], # hw -> wh
        'coord2_ec': soft_kpts1_c[..., [1, 0]],
    }

    return coarse_matches

def get_fine_match(fine_module, coords_normed, data):
    W, WW, C, scale = fine_module.W, fine_module.WW, fine_module.C, fine_module.scale

    # mkpts0_f and mkpts1_f
    mkpts0_f = data['mkpts0_c']
    scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
    mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]

    return {
        "coord1_ef": mkpts0_f,
        "coord2_ef": mkpts1_f
    }

@hydra_instantiable(name="aspanformer", group="model/matching")
class ASpanFormerWrapper(torch.nn.Module):
    def __init__(self, config=None, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        self.matcher_type = "detector-free"
        # Misc
        if config is None:
            config = get_cfg_defaults()

        # use the indoor config
        config.ASPAN.COARSE.COARSEST_LEVEL = [15,20]
        config.ASPAN.COARSE.TRAIN_RES = [480,640]
        config.ASPAN.MATCH_COARSE.BORDER_RM = 0

        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['aspan'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = ASpanFormer(config=_config['aspan']).to('cpu')
        self.loss = ASpanLoss(_config)

        # Pretrained weights
        print(pretrained_ckpt)
        if pretrained_ckpt:
            print('load')
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            msg=self.matcher.load_state_dict(state_dict, strict=False)
            print(msg)
            print(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir

        # # freeze layer
        # self.matcher.backbone.requires_grad_(False)
        # self.matcher.coarse_matching.requires_grad_(False)
        # self.matcher.loftr_coarse.requires_grad_(False)

    def forward(self, img_r=None, img_n=None, apply_mask=True, **kwargs):
        # to gray scale
        img_r = img_r.mean(dim=1, keepdim=True)
        img_n = img_n.mean(dim=1, keepdim=True)
        data = dict(image0=img_r, image1=img_n)

        self.matcher.coarse_matching.training = False
        self.matcher(data)
        # soft_coarse = soft_coarse_match(self.matcher.coarse_matching, data['conf_matrix'], data)
        soft_fine = get_fine_match(self.matcher.fine_matching, data['expec_f'][:, :2], data)
        # data.update(soft_coarse)
        data.update(soft_fine)
        mconf = data['mconf']
        m_bids = data['m_bids']

        mconf_mask = mconf > 0.2
        target_len_per_batch = 2000

        keys = [ 'coord1_ef', 'coord2_ef']  # 'coord1_ec', 'coord2_ec',
        per_batch = {k: [] for k in keys}
        n_corrs = []
        for bs in range(data['bs']):
            mask = m_bids == bs
            mask = torch.logical_and(mask, mconf_mask)
            if target_len_per_batch is not None and mask.sum() > target_len_per_batch:
                # select top k of mconf_mask
                _, top_indices = torch.topk(mconf * mask, target_len_per_batch)
                mask[...] = False
                mask[top_indices] = True

            for i, key in enumerate(keys):
                val = data[key][mask]
                n_corr = len(val)
                if i == 0:
                    n_corrs.append(n_corr)
                else:
                    assert n_corr == n_corrs[-1]
                per_batch[key].append(val)

        for key in per_batch:
            per_batch[key], lens = pad_to_same(per_batch[key], max_len=target_len_per_batch)
            assert lens == n_corrs
        matches = torch.cat([per_batch['coord1_ef'], per_batch['coord2_ef']], dim=-1)
        n_corr = max(n_corrs)
        match_idx_half = torch.arange(n_corr).to(matches.device)
        match_idx = match_idx_half.expand(2, n_corr).T # [[0, 0], [1, 1], ...]            
        match_indices = []
        for i in range(len(matches)):
            match_indices.append(match_idx.clone())
        per_batch['coord1_ec'] = per_batch['coord1_ef']
        per_batch['coord2_ec'] = per_batch['coord2_ef']
        return match_indices, matches, n_corrs, dictseq2seqdict(per_batch)