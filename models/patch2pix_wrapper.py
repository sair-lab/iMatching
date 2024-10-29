# code for evaluation
# https://github.com/GrumpyZhou/patch2pix/blob/bcbc0a034cd956f8d76d8fb014cdd7a8597efc36/utils/eval/model_helper.py#L64

# code for training
# https://github.com/GrumpyZhou/patch2pix/blob/bcbc0a034cd956f8d76d8fb014cdd7a8597efc36/train_patch2pix.py#L94

# jypyter notebook demo
# https://github.com/GrumpyZhou/patch2pix/blob/main/examples/visualize_matches.ipynb

# the model init script in the demo for eval
# https://github.com/GrumpyZhou/patch2pix/blob/bcbc0a034cd956f8d76d8fb014cdd7a8597efc36/utils/eval/model_helper.py#L32

# image res: eth3d 739 / 458 = 1.61
# dataloader res: 480 / 360 = 1.333

# trim landmark
# https://vscode.dev/github/Xanthorapedia/slamsup/blob/master/vo/mapping/bundle_adjustment.py#L48

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from attr import dataclass

from utils.config import hydra_config, hydra_instantiable
from utils.misc import dictseq2seqdict

from ext.patch2pix.networks.patch2pix import Patch2Pix
from argparse import Namespace
import torchvision.transforms as transforms
from ext.patch2pix.networks.utils import select_local_patch_feats, filter_coarse, sampson_dist


def load_weights(weights_dir, device):
    map_location = lambda storage, loc: storage.cuda(device.index) if False else storage
    weights_dict = None
    if weights_dir is not None: 
        weights_dict = torch.load(weights_dir, map_location=map_location)
    return weights_dict

def load_ckpt(ckpt_path, config, resume=False):    
    # Fine matching dist and io , qt_err, pass_rate 
    best_vals = [np.inf, 0.0, np.inf, 0.0]
    ckpt = load_weights(ckpt_path, config.device)
    
    if 'backbone' in ckpt:
        config.feat_idx = ckpt['feat_idx']
        config.weights_dict = ckpt['state_dict']
        config.backbone = ckpt['backbone']
        config.regressor_config = ckpt['regressor_config']
        if 'change_stride' in ckpt:
            config.change_stride = ckpt['change_stride']        
                
        if resume:
            config.start_epoch = ckpt['last_epoch'] + 1            
            config.optim_config.optimizer_dict = ckpt['optim']
            if 'lr_scheduler' in ckpt:
                config.optim_config.lr_scheduler_dict = ckpt['lr_scheduler']  

            if 'best_vals' in ckpt:
                if len(ckpt['best_vals']) == len(best_vals):
                    best_vals = ckpt['best_vals']
    
    else:
        # Only the pretrained weights
        config.weights_dict = ckpt        
    return best_vals

def init_model_config(args, lprint_):
    """This is a quick wrapper for model initialization
    Currently support method = patch2pix / ncnet.
    """
    
     # Initialize model
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')    
    regressor_config = Namespace(conv_dims=args.conv_dims,
                                 conv_kers=args.conv_kers,
                                 conv_strs=args.conv_strs,
                                 fc_dims=args.fc_dims,
                                 feat_comb=args.feat_comb,
                                 psize=args.psize,
                                 pshift = args.pshift,
                                 panc = args.panc,
                                 shared = args.shared)
    
    optim_config = Namespace(opt='adam',
                             lr_init=args.lr_init,
                             weight_decay=args.weight_decay,
                             lr_decay=args.lr_decay,
                             optimizer_dict=None,
                             lr_scheduler_dict=None)
    
    config = Namespace(training=True,
                       start_epoch=0,
                       device=device,
                       regr_batch=args.regr_batch,
                       backbone=args.backbone,
                       freeze_feat=args.freeze_feat,
                       change_stride=args.change_stride,
                       feat_idx=args.feat_idx,
                       regressor_config=regressor_config, 
                       weights_dict=None,
                       optim_config=optim_config) 
        
    
    # Fine matching dist and io , qt_err, pass_rate
    best_vals = [np.inf, 0.0, np.inf, 0.0]    
    if args.resume:
        # Continue training
        ckpt = os.path.join(args.out_dir, 'last_ckpt.pth')
        if os.path.exists(ckpt):
            args.ckpt = ckpt
    if args.pretrain:
        # Initialize with pretrained nc
        best_vals = load_ckpt(args.pretrain, config)
        lprint_('Load pretrained: {}  vals: {}'.format(args.pretrain, best_vals))        
    if args.ckpt:
        # Load a specific model
        best_vals = load_ckpt(args.ckpt, config, resume=args.resume)
        lprint_('Load model: {}  vals: {}'.format(args.ckpt, best_vals))
        
    return config, best_vals


@hydra_config(name="patch2pix_cfg_default", group="model/matching/patch2pix_config")
@dataclass
class Patch2PixConfig:
    gpu:int = 0 
    seed:int = 1     
    epochs:int = 100 
    save_step:int = 1 
    plot_counts:int = 5    
    batch:int = 8 
    regr_batch:int = 1200
    visdom_host:Optional[str] = None  
    visdom_port:Optional[str] = None      
    prefix:str = ''    
    out_dir:str = 'output/patch2pix'  
    
    # Data loading config
    dataset:str = 'MegaDepth'
    data_root:str = 'data'
    pair_root:str = 'data_pairs'
    match_npy:str = 'megadepth_pairs.ov0.35_imrat1.5.pair500.excl_test.npy'

    # Model architecture
    backbone:str = 'ResNet34'
    change_stride:bool = True  # from best train args
    ksize:int = 2
    freeze_feat:int = 87    
    feat_idx:list[int] = [0, 1, 2, 3]    
    feat_comb:str = 'pre'
    conv_kers:list[int] = [3, 3]
    conv_dims:list[int] = [512, 512]
    conv_strs:list[int] = [2, 1]
    fc_dims:list[int] = [512, 256]    
    psize:list[int] = [16, 16]
    pshift:int = 8
    panc:int = 8  # same as best args  
    ptmax:int = 250  # 400 (default) -> 250 due to GPU memory
    shared:bool = False    
    
    # Matching thresholds
    cthres:float = 0.5
    cls_dthres:list[int] = [50, 5]
    epi_dthres:list[int] = [50, 5]

    # Model intialize
    pretrain:Optional[str] = None   
    ckpt:Optional[str] = None                            
    resume:bool = False  # Auto load last cpkt

    # Optimization
    lr_init:float = 5e-4 
    lr_decay:Optional[list] = None # Opt: 'step' 'multistep'
    weight_decay:float = 0 
    weight_cls:float = 10.0
    weight_epi:list = [1, 1]

@hydra_instantiable(name="patch2pix", group="model/matching")
class Patch2PixWrapper(nn.Module):
    def __init__(self,
                 patch2pix_config: Patch2PixConfig = Patch2PixConfig(),
                 model_path:Optional[str]=None,
                 ransac_tol: float = -1.) -> None:
        super().__init__()
        self.matcher_type = "detector-free"
        if model_path is not None:
            patch2pix_config.ckpt = 'pretrained/patch2pix_pretrained.pth'
        model_config, best_vals = init_model_config(patch2pix_config, str)
        model_config.device = 'cpu'
        tmp = torch.__version__
        torch.__version__ = '1.'
        self.model = Patch2Pix(model_config)
        torch.__version__ = tmp

        self.config = patch2pix_config
        self.ransac_tol = ransac_tol
        self.transform = transforms.Compose([
                                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225)),])
        self.model.geo_dist_fn = sampson_dist
    
    def to(self, device):
        raise NotImplementedError

    def forward(self, img_r=None, img_n=None, apply_mask=True, **kwargs):
        img_r = self.transform(img_r)
        img_n = self.transform(img_n)
        im_src, im_pos = img_r, img_n

        net = self.model
        args = self.config
        ksize = args.ksize
        
        # Estimate patch-level matches 
        corr4d, delta4d, feats1, feats2 = net.forward(im_src, im_pos, ksize=ksize, return_feats=True)
        coarse_matches, match_scores = net.cal_coarse_matches(corr4d, delta4d, ksize=ksize, upsample=net.upsample, center=True)
        
        if net.panc > 1 and args.ptmax > 0:
            coarse_matches, match_scores = filter_coarse(coarse_matches, match_scores, 0.0, True, ptmax=args.ptmax)        

        # Coarse matches to locate anchors    
        coarse_matches = net.shift_to_anchors(coarse_matches)

        # Mid level matching on positive pairs
        mid_matches, mid_probs = net.forward_fine_match(feats1, feats2, 
                                                        coarse_matches, 
                                                        psize=net.psize[0],
                                                        ptype=net.ptype[0],
                                                        regressor=net.regress_mid)
        
        # Fine level matching based on mid matches
        fine_matches, fine_probs = net.forward_fine_match(feats1, feats2, 
                                                          mid_matches,               
                                                          psize=net.psize[1],  
                                                          ptype=net.ptype[1],
                                                          regressor=net.regress_fine)
        coarse_matches = torch.stack(coarse_matches)
        mid_matches = torch.stack(mid_matches)
        fine_matches = torch.stack(fine_matches)
        fine_probs = torch.stack(fine_probs)
        aux_info = {
            'coord1_ec': mid_matches[..., :2],
            'coord2_ec': mid_matches[..., 2:],
            'coord1_ef': fine_matches[..., :2],
            'coord2_ef': fine_matches[..., 2:],
        }  # TODO: maybe use the probability during error calculation
        n_corrs = [fine_probs[i].shape[0] for i in range(len(fine_probs))]
        n_corr = max(n_corrs)
        match_idx_half = torch.arange(n_corr).to(fine_matches.device)
        match_idx = match_idx_half.expand(2, n_corr).T # [[0, 0], [1, 1], ...]
        match_indices, match_masks = [], []
        for i in range(len(fine_matches)):
            match_indices.append(match_idx.clone())
        
        def masked(x, masks):
            n_corrs = []
            if x.dtype is torch.int32 or \
                x.dtype is torch.int64:
                fill_val = -1
            else:
                fill_val = torch.nan
            filtered = x.new_full(size=x.shape, fill_value=fill_val)
            
            for i in range(len(x)):
                mask = masks[i]
                n_corr = mask.sum()
                n_corrs.append(n_corr)
                filtered[i, :n_corr] = x[i][mask]
            return filtered, n_corrs
        if apply_mask:
            masks = fine_probs < args.cthres
            filtered_matches, n_corrs = masked(fine_matches, masks)
            for k in aux_info:
                aux_info[k], _ = masked(aux_info[k], masks)
            # the original keys used by loss
            for_loss = dict(coarse_matches=coarse_matches, 
                            mid_matches=mid_matches, 
                            fine_matches=fine_matches, 
                            mid_probs=mid_probs, 
                            fine_probs=fine_probs,)
            aux_info.update(for_loss)
            return match_indices, filtered_matches, n_corrs, dictseq2seqdict(aux_info)
    
    def loss(self, Fs, coarse_matches, mid_matches, fine_matches, mid_probs, fine_probs):
        skipped = 0
        # Calculate per pair losses
        net = self.model
        args = self.config
        cthres, cls_dthres, epi_dthres = args.cthres, args.cls_dthres, args.epi_dthres
        cls_loss_weight = args.weight_cls
        efine_weight, emid_weight = args.weight_epi

        cls_batch_lss = []
        epi_batch_lss = []
        for F, cmat, mmat, fmat, mcls_pred, fcls_pred in zip(Fs, coarse_matches, 
                                                             mid_matches, fine_matches, 
                                                             mid_probs, fine_probs):
            N = len(cmat)
            
            # Classification gt based on coarse matches             
            cdist = net.geo_dist_fn(cmat, F)
            mdist = net.geo_dist_fn(mmat, F)
            fdist = net.geo_dist_fn(fmat, F)
            ones = torch.ones_like(cdist)
            zeros = torch.zeros_like(cdist)

            # Classification loss
            mcls_pos = torch.where(cdist < cls_dthres[0], ones, zeros)
            fcls_pos = torch.where(mdist < cls_dthres[1], ones, zeros)
            mcls_neg = 1 - mcls_pos
            fcls_neg = 1 - fcls_pos
            
            if mcls_pos.sum() == 0 or fcls_pos.sum() == 0:
                skipped += 1
                continue
                        
            mcls_weights = mcls_neg.sum() / mcls_pos.sum() * mcls_pos + mcls_neg 
            mcls_lss = nn.functional.binary_cross_entropy(mcls_pred, mcls_pos, reduction='none')
            mcls_lss = (mcls_weights * mcls_lss).mean()

            fcls_weights = fcls_neg.sum() / fcls_pos.sum() * fcls_pos + fcls_neg 
            fcls_lss = nn.functional.binary_cross_entropy(fcls_pred, fcls_pos, reduction='none')
            fcls_lss = (fcls_weights * fcls_lss).mean()            
                
            cls_lss = mcls_lss + fcls_lss    
            cls_batch_lss.append(cls_lss)
                        
            # Epipolar loss
            mids_gt = torch.where(cdist < epi_dthres[0], ones, zeros).nonzero(as_tuple=False).flatten()
            fids_gt = torch.where(mdist < epi_dthres[1], ones, zeros).nonzero(as_tuple=False).flatten()
            #lprint_(f'{len(mdist)} {len(mids_gt)} {len(fdist)} {len(fids_gt)}')
            
            if len(fids_gt) == 0 and len(mids_gt) == 0: 
                skipped += 1
                continue

            epi_mid = mdist[mids_gt].mean() if len(mids_gt) > 0 else torch.tensor(0).to(mdist)
            epi_fine = fdist[fids_gt].mean() if len(fids_gt) > 0 else torch.tensor(0).to(fdist)
            epi_lss = emid_weight * epi_mid + efine_weight * epi_fine
            epi_batch_lss.append(epi_lss)                        

        # Total loss
        cls_loss = torch.stack(cls_batch_lss).mean() if len(cls_batch_lss) > 0 else torch.tensor(0.0, requires_grad=True).to(net.device)
        epi_loss = torch.stack(epi_batch_lss).mean() if len(epi_batch_lss) > 0 else torch.tensor(0.0, requires_grad=True).to(net.device)        
        loss = cls_loss_weight * cls_loss + epi_loss

        return loss