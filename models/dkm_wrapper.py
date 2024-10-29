from typing import Optional
import torch
import torch.nn.functional as F

from utils.config import hydra_config, hydra_instantiable
import ext.DKM.dkm.models.model_zoo as dkm_model_zoo
import ext.RoMa.roma.models.model_zoo as roma_model_zoo
from ext.DKM.dkm.utils import get_tuple_transform_ops
from utils.misc import dictseq2seqdict, pad_to_same



@hydra_instantiable(name="dkm", group="model/matching")
class dkmWrapper(torch.nn.Module):
    def __init__(self, model_zoo:str='indoor', model_path:Optional[str]=None):
        self.matcher_type = "detector-free"
        super().__init__()
        if model_zoo == 'indoor':
            self.model = dkm_model_zoo.DKMv3_indoor(path_to_weights=model_path)
            self.flow_key = 'dense_flow'
            self.certainty_key = "dense_certainty"
        elif model_zoo == 'outdoor':
            self.model = dkm_model_zoo.DKMv3_outdoor(path_to_weights=model_path)
            self.flow_key = 'dense_flow'
            self.certainty_key = "dense_certainty"
        elif model_zoo == 'roma_indoor':
            self.model = roma_model_zoo.roma_indoor(device='cpu', path_to_weights=model_path)
            self.flow_key = 'flow'
            self.certainty_key = "certainty"
        elif model_zoo == 'roma_outdoor':
            self.model = roma_model_zoo.roma_outdoor(device='cpu', path_to_weights=model_path)
            self.flow_key = 'flow'
            self.certainty_key = "certainty"
        else:
            raise ValueError
        self.append_coarse = False
        self.interp_coarse = True
        self.connected_sampling = False
        
        def freeze_module(m):
            for param in m.parameters():
                param.requires_grad = False
        freeze_coarse = True
        if freeze_coarse:
            freeze_module(self.model.encoder)
            freeze_module(self.model.decoder.gps)
            freeze_module(self.model.decoder.proj)
            freeze_module(self.model.decoder.embedding_decoder)
        pass

    def sample(
        self,
        dense_matches,
        dense_certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            dense_certainty = dense_certainty.clone()
            dense_certainty[dense_certainty > upper_thresh] = 1
        elif "pow" in self.sample_mode:
            dense_certainty = dense_certainty**(1/3)
        elif "naive" in self.sample_mode:
            dense_certainty = torch.ones_like(dense_certainty)
        matches, certainty = (
            dense_matches.reshape(-1, dense_matches.shape[-1]),
            dense_certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty

        from dkm.utils.kde import kde
        density = kde(good_matches, std=0.1, device='cpu')
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return balanced_samples

    def parse_output(self, selected_scale, dense_corresps):
        symmetric = False
        device = dense_corresps[1][self.flow_key].device
        b = dense_corresps[1][self.flow_key].shape[0]
        # finest_scale = 1 or 2 4 8 16 32 
        query_to_support = dense_corresps[selected_scale][self.flow_key]
        flow_h, flow_w = query_to_support.shape[-2:]
        low_res_certainty = F.interpolate(
            dense_corresps[16][self.certainty_key], size=(flow_h, flow_w), align_corners=False, mode="bilinear"
        )
        cert_clamp = 0
        factor = 0.5 if selected_scale < 16 else 0
        low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)

        if self.upsample_preds: 
            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True
            )
            query, support = test_transform((im1, im2))
            query, support = query[None].to(device), support[None].to(device)
            batch = {"query": query, "support": support, "corresps": dense_corresps[selected_scale]}
            if symmetric:
                dense_corresps = self.forward_symmetric(batch, upsample = True, batched=True)
            else:
                dense_corresps = self.forward(batch, batched = True, upsample=True)
        dense_certainty = dense_corresps[selected_scale][self.certainty_key]
        
        # Get certainty interpolation
        dense_certainty = dense_certainty - low_res_certainty
        query_to_support = query_to_support.permute(0, 2, 3, 1)
        # Create im1 meshgrid
        query_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / flow_h, 1 - 1 / flow_h, flow_h, device=device),
                torch.linspace(-1 + 1 / flow_w, 1 - 1 / flow_w, flow_w, device=device),
            ),
            indexing = 'ij',
        )
        query_coords = torch.stack((query_coords[1], query_coords[0]))
        query_coords = query_coords[None].expand(b, 2, flow_h, flow_w)
        dense_certainty = dense_certainty.sigmoid()  # logits -> probs
        query_coords = query_coords.permute(0, 2, 3, 1)
        if (query_to_support.abs() > 1).any() and True:
            wrong = (query_to_support.abs() > 1).sum(dim=-1) > 0
            dense_certainty[wrong[:,None]] = 0
            
        query_to_support = torch.clamp(query_to_support, -1, 1)
        if symmetric:
            support_coords = query_coords
            qts, stq = query_to_support.chunk(2)                    
            q_warp = torch.cat((query_coords, qts), dim=-1)
            s_warp = torch.cat((stq, support_coords), dim=-1)
            warp = torch.cat((q_warp, s_warp),dim=2)
            dense_certainty = torch.cat(dense_certainty.chunk(2), dim=3)[:,0]
        else:
            warp = torch.cat((query_coords, query_to_support), dim=-1)
        return warp, dense_certainty

    @staticmethod
    def pair_list_of_batch(b):
        total_frame = 0
        n_ref = 2
        pair_infos = []
        try:
            pair_infos = []
            while not len(pair_infos) == b:
                for i in range(total_frame):
                    if total_frame - i <= n_ref:
                        pair_infos.append([i, total_frame])
                total_frame += 1
                if len(pair_infos) > b:
                    raise ValueError
        except ValueError:
            total_frame = b // n_ref
            pair_infos = []
            for i in range(total_frame + n_ref):
                for j in range(i - n_ref, i):
                    pair_infos.append([i, j])
        return pair_infos, total_frame

    def select_on_pair(self, dense_certainty, num_selected=5000, training=True):
        b = len(dense_certainty)
        # collect pair info
        if training:
            pair_infos, total_frame = self.pair_list_of_batch(b)
        else:
            pair_infos = [(i, i+1) for i in range(b)]
            total_frame = b + 1
        validity = [None] * b
        dense_certainty_cpu = dense_certainty.cpu().float()
        for frame in range(total_frame): 
            filtered_pair_indices = []
            for b_idx, pair in enumerate(pair_infos):
                if pair[0] == frame:
                    filtered_pair_indices.append(b_idx)
            if len(filtered_pair_indices) == 0:
                continue
            certainty = dense_certainty_cpu[filtered_pair_indices].mean(0).flatten()
            # selected = self.sample(match, certainty, )
            # selected = torch.logical_and(torch.rand_like(certainty) < 0.08, certainty > 0.1)
            valid_indices = (certainty.sigmoid() > self.sample_thresh).nonzero().flatten()
            selected_indices = torch.randperm(len(valid_indices))
            if num_selected is not None:
                selected_indices = selected_indices[:num_selected]
            selected = valid_indices[selected_indices]

            for b_idx in filtered_pair_indices:
                assert validity[b_idx] is None
                validity[b_idx] = selected
        return validity

    def interp_sampling(self, dense_corresps, col_coarse_0, i):
        sampled_fine = torch.nn.functional.grid_sample(dense_corresps[1][self.flow_key][i][None], 
                                                        col_coarse_0[None][None], align_corners=False)
        with torch.no_grad():
            sampled_conf = torch.nn.functional.grid_sample(dense_corresps[1][self.certainty_key][i][None], 
                                                        col_coarse_0[None][None], align_corners=False)
        sampled_fine = sampled_fine[0, :, 0, :].mT
        sampled_conf = sampled_conf[0, :, 0, :].mT.squeeze()
        return sampled_fine, sampled_conf

    def forward(self, img_r=None, img_n=None, apply_mask=True, **kwargs):
        self.model.encoder.eval()
        self.model.decoder.gps.eval()
        self.model.decoder.proj.eval()
        self.model.decoder.embedding_decoder.eval()

        ws = self.model.w_resized
        hs = self.model.h_resized
        original_shape = (img_r.shape[-1], img_r.shape[-2])
        test_transform = get_tuple_transform_ops(
            resize=(hs, ws), normalize=True
            )
        with torch.no_grad():
            img_r, img_n = test_transform((img_r, img_n))
        batch = dict(query=img_r, support=img_n, im_A=img_r, im_B=img_n)

        dense_corresps = self.model(batch)
        b = img_r.shape[0]
        ws = self.w_resized = original_shape[0]
        hs = self.h_resized = original_shape[1]
        self.upsample_preds = False
        if self.upsample_preds:
            hs, ws = self.upsample_res
        self.sample_mode = "threshold_balanced"
        thres = self.sample_thresh = 0.15 # TODO

        if self.training and not self.append_coarse and not self.interp_coarse:
            warp, dense_certainty = self.parse_output(1, dense_corresps)
            kpts_A, kpts_B = self.model.to_pixel_coordinates(warp, hs, ws, hs, ws)
            kpts_per_batch = torch.cat([kpts_A, kpts_B], dim=-1)

            # for tracker
            matches = []
            n_corrs = []
            with torch.no_grad():
                validity = self.select_on_pair(dense_certainty, num_selected=5000)
            
            for i, (kpts, conf) in enumerate(zip(kpts_per_batch, dense_certainty[:, 0])):
                matches.append(kpts.flatten(0, 1)[validity[i]])
                n_corrs.append(len(matches[-1]))
        elif self.training and self.append_coarse:
            selected_levels = [i for i in dense_corresps.keys() if i <= 8]
            selected_per_level = {k:5000//k for k in selected_levels}
            # for tracker
            matches = {}
            n_corrs = []
            for level in selected_levels:
                warp, dense_certainty = self.parse_output(level, dense_corresps)
                kpts_A, kpts_B = self.model.to_pixel_coordinates(warp, hs, ws, hs, ws)
                kpts_per_batch = torch.cat([kpts_A, kpts_B], dim=-1)

                with torch.no_grad():
                    validity = self.select_on_pair(dense_certainty, num_selected=selected_per_level[level]) # selected_per_level[level]
                matches[level] = []
                for i, (kpts, conf) in enumerate(zip(kpts_per_batch, dense_certainty[:, 0])):
                    matches[level].append(kpts.flatten(0, 1)[validity[i]])
            matches_ = []
            for i in range(b):
                matches_.append(torch.cat([matches[k][i] for k in matches]))
                n_corrs.append(len(matches_[-1]))
            matches = matches_
        elif self.training and self.interp_coarse:
            # randomly sample a layer
            selected_levels = [i for i in dense_corresps.keys() if i >= 1 and i <=16]
            selected_per_level = {k:2500//k for k in selected_levels}
            # for tracker
            matches_c = {}
            n_corrs = []
            for level in selected_levels:
                warps, dense_certainty = self.parse_output(level, dense_corresps)
                # kpts_A, kpts_B = self.model.to_pixel_coordinates(warp, hs, ws, hs, ws)
                # kpts_per_batch = torch.cat([kpts_A, kpts_B], dim=-1)

                with torch.no_grad():
                    validity = self.select_on_pair(dense_certainty, num_selected=selected_per_level[level], training=self.training) # selected_per_level[level]
                matches_c[level] = []
                for i, (kpts, conf) in enumerate(zip(warps, dense_certainty[:, 0])):
                    matches_c[level].append(kpts.flatten(0, 1)[validity[i]])
            if self.connected_sampling:
                pair_infos, total_frame = self.pair_list_of_batch(b)
                point_buckets = [list() for _ in range(total_frame)]
                matches_c[-1] = [None, ] * b
                for i in range(total_frame - 1):
                    # i -> i+1, look for anchor points in i+1
                    b_idx = pair_infos.index([i, i+1])
                    flow = matches_c[1][b_idx]
                    point_buckets[i+1].append(flow[:, 2:])
                    if len(point_buckets[i]) > 0:
                        anchors = torch.cat(point_buckets[i])
                        sampled_fine, sampled_conf = self.interp_sampling(dense_corresps, anchors, i)
                        mask = sampled_conf.sigmoid() > 0.15
                        matches_c[-1][b_idx] = torch.cat([anchors, sampled_fine], dim=-1)[mask]
                        point_buckets[i+1].append(sampled_fine[mask])
                # add as -1 key
            matches_c_ = []
            matches = []
            for i in range(b):
                col_coarse = torch.cat([matches_c[k][i] for k in matches_c if matches_c[k][i] is not None])
                # with this, sample on finest level
                col_coarse_0 = col_coarse[..., :2]
                sampled_fine, sampled_conf = self.interp_sampling(dense_corresps, col_coarse_0, i)
                sampled_fine = torch.cat([col_coarse_0, sampled_fine], dim=-1)
                mask = sampled_conf.sigmoid() > self.sample_thresh
                col_coarse = col_coarse[mask]
                matches_c_.append(torch.cat(self.model.to_pixel_coordinates(col_coarse, hs, ws, hs, ws), dim=-1))
                n_corrs.append(len(col_coarse))
                sampled_fine = sampled_fine[mask]
                sampled_fine = torch.cat(self.model.to_pixel_coordinates(sampled_fine, hs, ws, hs, ws), dim=-1)
                matches.append(sampled_fine)
                assert matches[-1].shape == matches_c_[-1].shape
            matches_c = matches_c_
            matches_c, _ = pad_to_same(matches_c)
        else:
            warp, dense_certainty = self.parse_output(1, dense_corresps)
            kpts_A, kpts_B = self.model.to_pixel_coordinates(warp, hs, ws, hs, ws)
            kpts_per_batch = torch.cat([kpts_A, kpts_B], dim=-1)

            # for tracker
            matches = []
            n_corrs = []

            for i, (kpts, conf) in enumerate(zip(kpts_per_batch, dense_certainty[:, 0])):
                matches.append(kpts[conf > thres])
                n_corrs.append(len(matches[-1]))

        matches, lens = pad_to_same(matches)
        n_corr = max(n_corrs)
        match_idx_half = torch.arange(n_corr).to(matches.device)
        match_idx = match_idx_half.expand(2, n_corr).T # [[0, 0], [1, 1], ...]            
        match_indices = []
        for i in range(len(matches)):
            match_indices.append(match_idx.clone())
        if self.training and self.interp_coarse:
            per_batch = dict(coord1_ef=matches[..., :2], coord2_ef=matches[..., 2:], coord1_ec=matches_c[..., :2], coord2_ec=matches_c[..., 2:])
        else:
            per_batch = dict(coord1_ef=matches[..., :2], coord2_ef=matches[..., 2:], coord1_ec=matches[..., :2], coord2_ec=matches[..., 2:])
        return match_indices, matches, n_corrs, dictseq2seqdict(per_batch)
