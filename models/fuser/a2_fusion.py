import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class A2Fusion(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.key_feats = model_cfg.KEY_FEATS
        dim_feats = model_cfg.DIM_FEATS

        to_embed_method = model_cfg.TO_EMBED.METHOD
        dim_common = model_cfg.TO_EMBED.DIM_COMMON
        to_embed_activation = model_cfg.TO_EMBED.ACTIVATION
        to_embed_n_layer = model_cfg.TO_EMBED.N_LAYER
        to_embed_is_layer_norm = model_cfg.TO_EMBED.LAYER_NORM

        x_shape, y_shape, z_shape = grid_size

        ucp_method  = model_cfg.UCP.METHOD
        n_query = model_cfg.UCP.N_QUERY
        patch_y, patch_x = model_cfg.UCP.PATCH_SIZE
        dim_patch = model_cfg.UCP.DIM_PATCH
        ucp_activation = model_cfg.UCP.ACTIVATION
        ucp_n_layer = model_cfg.UCP.N_LAYER
        ucp_is_layer_norm = model_cfg.UCP.LAYER_NORM

        assert n_query % (patch_x*patch_y) == 0, '* # of queries should be divided by patch**2'
        
        # consider MHA -> Transformer
        n_heads = model_cfg.N_HEADS_MHA
        
        self.aware_query = nn.Parameter(torch.randn(1, n_query, dim_patch), requires_grad=True)
        for idx_key, temp_key in enumerate(self.key_feats):
            ### [To same dimension] ###
            if to_embed_method == 'None': # Ablation
                dim_common = dim_feats[idx_key]
                setattr(self, f'to_embed_{temp_key}', nn.Identity())
            elif to_embed_method == 'Linear1': # Default
                if to_embed_is_layer_norm:
                    setattr(self, f'to_embed_{temp_key}',
                            nn.Sequential(
                                Rearrange('b c y x -> b (y x) c'),
                                nn.LayerNorm(dim_feats[idx_key]),
                                nn.Linear(dim_feats[idx_key], dim_common, bias=False),
                                nn.LayerNorm(dim_common),
                                Rearrange('b (y x) c -> b c y x', y=y_shape, x=x_shape)
                            ))
                else:
                    setattr(self, f'to_embed_{temp_key}', nn.Conv2d(dim_feats[idx_key], dim_common, kernel_size=1, bias=False))
            elif to_embed_method == 'LinearN':
                if to_embed_is_layer_norm:
                    list_layers = [Rearrange('b c y x -> b (y x) c'), nn.LayerNorm(dim_feats[idx_key]), nn.Linear(dim_feats[idx_key], dim_common, bias=False)]
                    for _ in range(to_embed_n_layer-1):
                        list_layers.append(getattr(nn, to_embed_activation)())
                        list_layers.append(nn.Linear(dim_common, dim_common, bias=False))
                    list_layers.extend([nn.LayerNorm(dim_common), Rearrange('b (y x) c -> b c y x', y=y_shape, x=x_shape)])
                else: # wo LayerNorm
                    list_layers = [Rearrange('b c y x -> b (y x) c'), nn.Linear(dim_feats[idx_key], dim_common, bias=False)]
                    for _ in range(to_embed_n_layer-1):
                        list_layers.append(getattr(nn, to_embed_activation)())
                        list_layers.append(nn.Linear(dim_common, dim_common, bias=False))
                    list_layers.append(Rearrange('b (y x) c -> b c y x', y=y_shape, x=x_shape))
                setattr(self, f'to_embed_{temp_key}', nn.Sequential(*list_layers))
            ### [To same dimension] ###
            
            ### [Unified canonical projection] ###
            setattr(self, f'to_patch_{temp_key}', Rearrange('b c (y py) (x px) -> (b y x) (py px c)', py=patch_y, px=patch_x))
            dim_patch_input = patch_y*patch_x*dim_common
            if ucp_method == None:
                setattr(self, f'to_patch_embed_{temp_key}', nn.Identity())
            elif ucp_method == 'Linear1': # Default
                if ucp_is_layer_norm:
                    setattr(self, f'to_patch_embed_{temp_key}',
                            nn.Sequential(
                                nn.LayerNorm(dim_patch_input),
                                nn.Linear(dim_patch_input, dim_patch, bias=False), # Note: bias=True for default code
                                nn.LayerNorm(dim_patch)
                            ))
                else:
                    setattr(self, f'to_patch_embed_{temp_key}', nn.Linear(dim_patch_input, dim_patch))
            elif ucp_method == 'LinearN':
                if ucp_is_layer_norm:
                    list_layers = [nn.LayerNorm(dim_patch_input), nn.Linear(dim_patch_input, dim_patch, bias=False)]
                    for _ in range(ucp_n_layer-1):
                        list_layers.append(getattr(nn, ucp_activation)())
                        list_layers.append(nn.Linear(dim_patch, dim_patch, bias=False))
                    list_layers.append(nn.LayerNorm(dim_patch))
                else:
                    list_layers = [nn.Linear(dim_patch_input, dim_patch, bias=False)]
                    for _ in range(ucp_n_layer-1):
                        list_layers.append(getattr(nn, ucp_activation)())
                        list_layers.append(nn.Linear(dim_patch, dim_patch, bias=False))
                setattr(self, f'to_patch_embed_{temp_key}', nn.Sequential(*list_layers))
        
        self.fuser = nn.MultiheadAttention(dim_patch, n_heads, 0., batch_first=True)

        ### [Post feature transform] ###
        pft_method  = model_cfg.PFT.METHOD
        pft_activation = model_cfg.PFT.ACTIVATION
        ptf_n_layer = model_cfg.PFT.N_LAYER
        pft_is_layer_norm = model_cfg.PFT.LAYER_NORM

        if pft_method == 'None':
            self.pft = nn.Identity()
        elif pft_method == 'LN':
            self.pft = nn.LayerNorm(dim_patch)
        elif pft_method == 'Linear1':
            if pft_is_layer_norm:
                self.pft = nn.Sequential(
                    nn.LayerNorm(dim_patch_input),
                    nn.Linear(dim_patch, dim_patch, bias=False), # Note: bias=True for default code
                    nn.LayerNorm(dim_patch)
                )
            else:
                self.pft = nn.Linear(dim_patch, dim_patch)
        elif pft_method == 'LinearN':
            if pft_is_layer_norm:
                list_layers = [nn.LayerNorm(dim_patch), nn.Linear(dim_patch, dim_patch, bias=False)]
                for _ in range(ptf_n_layer-1):
                    list_layers.append(getattr(nn, pft_activation)())
                    list_layers.append(nn.Linear(dim_patch, dim_patch, bias=False))
                list_layers.append(nn.LayerNorm(dim_patch))
            else:
                list_layers = [nn.Linear(dim_patch, dim_patch, bias=False)]
                for _ in range(ptf_n_layer-1):
                    list_layers.append(getattr(nn, pft_activation)())
                    list_layers.append(nn.Linear(dim_patch, dim_patch, bias=False))
            self.pft = nn.Sequential(*list_layers)
        ### [Post feature transform] ###
        
        n_repeat_channel = int(round(n_query/(patch_x*patch_y)))
        is_post_channel = model_cfg.POST_CHANNEL

        if n_repeat_channel == 1:
            self.to_fused_feat = Rearrange('(b y x) (py px) c -> b c (y py) (x px)', y=int(y_shape/patch_y), x=int(x_shape/patch_x), py=patch_y, px=patch_x)
        else:
            if is_post_channel:
                self.to_fused_feat = Rearrange('(b y x) (py px ch) c -> b (c ch) (y py) (x px)', y=int(y_shape/patch_y), x=int(x_shape/patch_x), py=patch_y, px=patch_x, ch=n_repeat_channel)
            else:
                self.to_fused_feat = Rearrange('(b y x) (ch py px) c -> b (c ch) (y py) (x px)', y=int(y_shape/patch_y), x=int(x_shape/patch_x), py=patch_y, px=patch_x, ch=n_repeat_channel)

            ### Attention maps ###
            if is_post_channel:
                self.get_feat_w_channel = Rearrange('(b y x) (py px ch) c -> b c ch (y py) (x px)', y=int(y_shape/patch_y), x=int(x_shape/patch_x), py=patch_y, px=patch_x, ch=n_repeat_channel)
            else:
                self.get_feat_w_channel = Rearrange('(b y x) (ch py px) c -> b c ch (y py) (x px)', y=int(y_shape/patch_y), x=int(x_shape/patch_x), py=patch_y, px=patch_x, ch=n_repeat_channel)
            ### Attention maps ###

        ### [Sensor combination loss] ###
        self.is_scl = kwargs.get('scl', False)
        ### [Sensor combination loss] ###

    def forward(self, batch_dict):
        list_feats = []
        
        is_get_feats_to_vis = False
        if not self.training:
            if 'get_feats_to_vis' in batch_dict.keys():
                is_get_feats_to_vis = batch_dict['get_feats_to_vis']
                batch_dict['feat_b4_fusion'] = []

        for idx_key, temp_key in enumerate(self.key_feats):

            ### Adjusting availability ###
            if not self.training:
                if 'avail_feats' in batch_dict.keys():
                    if temp_key not in batch_dict['avail_feats']:
                        continue
            ### Adjusting availability ###

            # print(batch_dict[temp_key].shape)
            temp_feat = getattr(self, f'to_embed_{temp_key}')(batch_dict[temp_key])

            if is_get_feats_to_vis:
                batch_dict['feat_b4_fusion'].append(temp_feat)

            # print(temp_feat.shape)
            temp_feat = torch.unsqueeze(getattr(self, f'to_patch_{temp_key}')(temp_feat), dim=1)
            temp_feat = getattr(self, f'to_patch_embed_{temp_key}')(temp_feat)
            list_feats.append(temp_feat)

        kv_feats = torch.cat(list_feats, dim=1)
        b_patch, n_sensor, _ = kv_feats.shape
        q_feat = repeat(self.aware_query, 'b n c -> (b b_repeat) n c', b_repeat=b_patch)
        
        # print(q_feat.shape)
        # print(kv_feats.shape)

        if self.training:
            if self.is_scl:
                list_individual_feat = []
                for temp_kv_feat in list_feats: # per each sensor
                    list_individual_feat.append(self.to_fused_feat(self.pft(self.fuser(q_feat, temp_kv_feat, temp_kv_feat)[0])))
                # for sensor pair
                temp_arr = range(len(list_feats))
                temp_n = len(list_feats)
                for temp_i in range(temp_n):
                    for temp_j in range(temp_i + 1, temp_n):
                        # print(f"({temp_arr[temp_i]},{temp_arr[temp_j]})")
                        idx_pair_0 = temp_arr[temp_i]
                        idx_pair_1 = temp_arr[temp_j]

                        temp_kv_feat = torch.cat([list_feats[idx_pair_0], list_feats[idx_pair_1]], dim=1)
                        list_individual_feat.append(self.to_fused_feat(self.pft(self.fuser(q_feat, temp_kv_feat, temp_kv_feat)[0])))
                
                # print(len(list_individual_feat))
                batch_dict['list_individual_feat'] = list_individual_feat
        else:
            if 'feat_indiv' in batch_dict.keys():
                list_individual_feat = []
                for temp_kv_feat in list_feats: # per each sensor
                    list_individual_feat.append(self.to_fused_feat(self.pft(self.fuser(q_feat, temp_kv_feat, temp_kv_feat)[0])))
                batch_dict['feat_indiv'] = list_individual_feat

        ### Attention maps ###
        if 'get_att_maps' in batch_dict.keys():
            fused_feat, att_maps = self.fuser(q_feat, kv_feats, kv_feats)
            batch_dict['get_att_maps'] = self.get_feat_w_channel(att_maps)
        else:
            fused_feat = self.fuser(q_feat, kv_feats, kv_feats)[0]
        ### Attention maps ###

        if is_get_feats_to_vis:
            batch_dict['pre_fused_feat'] = self.to_fused_feat(fused_feat)
        
        fused_feat = self.pft(fused_feat)
        # print(fused_feat.shape)
        fused_feat = self.to_fused_feat(fused_feat)
        # print(fused_feat.shape)

        batch_dict['fused_feat'] = fused_feat

        return batch_dict
