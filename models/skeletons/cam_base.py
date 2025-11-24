import torch
import torch.nn as nn

from ops.bev_pool import bev_pool

from models import backbone_2d, head

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx

class CamBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.dataset_cfg = cfg.DATASET

        # class
        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)
        # print(self.class_names)
        
        ### Backbone ###
        self.backbone = backbone_2d.__all__[self.model_cfg.BACKBONE.NAME](self.cfg)
        is_freeze = self.model_cfg.BACKBONE.FREEZE
        is_freeze_bn = self.model_cfg.BACKBONE.FREEZE_BN
        if is_freeze or is_freeze_bn:
            self.backbone.freeze(is_freeze, is_freeze_bn)
        ### Backbone ###
        
        ### Considering FPN if we apply freeze ###
        # We need learnable parameters for soft depth estimation when freezing backbone
        self.neck = backbone_2d.__all__[self.model_cfg.NECK.NAME](self.model_cfg.NECK)
        ### Considering FPN if we apply freeze ###

        ### LSS ###
        in_channel = self.model_cfg.LSS.IN_CHANNEL
        out_channel = self.model_cfg.LSS.OUT_CHANNEL
        self.image_size = self.model_cfg.LSS.IMAGE_SIZE
        self.feature_size = self.model_cfg.LSS.FEATURE_SIZE
        xbound = self.model_cfg.LSS.XBOUND
        ybound = self.model_cfg.LSS.YBOUND
        zbound = self.model_cfg.LSS.ZBOUND
        self.dbound = self.model_cfg.LSS.DBOUND
        downsample = self.model_cfg.LSS.DOWNSAMPLE
        self.accelerate = self.model_cfg.get("ACCELERATE",False)
        if self.accelerate:
            self.cache = None

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = in_channel
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.depthnet = nn.Conv2d(in_channel, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            # map segmentation
            if self.model_cfg.LSS.get('USE_CONV_FOR_NO_STRIDE',False):
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(True),
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                )
            else:
                self.downsample = nn.Identity()

        self.head = head.__all__[self.model_cfg.HEAD.NAME](cfg=cfg)

        self.w_depth = self.model_cfg.LSS.LOSS.W_DEPTH
        self.w_distil = self.model_cfg.LSS.LOSS.W_DISTIL

        cfg_distil = self.model_cfg.LSS.LOSS.get('DISTIL_LOSS', None)

        if cfg_distil is not None:
            self.is_distil = True
            self.distil_loss_type = self.model_cfg.LSS.LOSS.DISTIL_LOSS
            self.mask_area = self.model_cfg.LSS.LOSS.MASK_AREA
            self.fb_rate = self.model_cfg.LSS.LOSS.FB_RATE
        else:
            self.is_distil = False

        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING

    def get_cam_feats(self, x):
        x = x.to(torch.float)
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1) # B, D, H, W
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        if self.training:
            depth = depth.view(B, N, self.D, fH, fW)
            return x, depth
        else:
            return x

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        lidar2img
    ):
        lidar2img = lidar2img.to(torch.float)
        
        B,N = lidar2img.shape[:2]
        D,H,W = self.frustum.shape[:3]
        points = self.frustum.view(1,1,D,H,W,3).repeat(B,N,1,1,1,1)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = torch.cat([points,torch.ones_like(points[...,-1:])],dim=-1)
        points = points.unsqueeze(-1)
        # points = torch.inverse(img_aug_matrix).view(B,N,1,1,1,4,4).matmul(points.unsqueeze(-1))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
                torch.ones_like(points[:, :, :, :, :, 2:3])
            ),
            5,
        )
        points = torch.inverse(lidar2img).view(B,N,1,1,1,4,4).matmul(points).squeeze(-1)[...,:3]

        return points

    def bev_pool(self, geom_feats, x):
        geom_feats = geom_feats.to(torch.float)
        x = x.to(torch.float)

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        if self.accelerate and self.cache is None:
            self.cache = (geom_feats,kept)

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def acc_bev_pool(self,x):
        geom_feats,kept = self.cache
        x = x.to(torch.float)

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)
        x = x[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(self, batch_dict):
        front_img = batch_dict['camera_imgs'][:,0,:,:,:].cuda()
        # print(batch_dict['sensor2image'])
        list_feat_img = self.backbone(front_img)
        batch_dict['image_features'] = list_feat_img
        batch_dict = self.neck(batch_dict)

        # for feat in list_feat_img:
        #     print(feat.shape)

        ### Vis feature map ###
        # import matplotlib.pyplot as plt
        # print(front_img.shape)
        # print(feat_img.shape)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # img = torch.squeeze(front_img.detach().cpu(), 0).permute(1, 2, 0).numpy()
        # ax1.imshow(img)
        # ax1.set_title('input')
        # feature = torch.mean(torch.squeeze(feat_img.detach().cpu(), 0), dim=0).numpy()
        # im = ax2.imshow(feature, cmap='viridis')
        # ax2.set_title('feature')
        # plt.colorbar(im, ax=ax2)
        # plt.tight_layout()
        # plt.show()
        ### Vis feature map ###

        x = batch_dict['image_fpn'] 
        if not isinstance(x, torch.Tensor):
            x = x[0] # BN, 256, 32, 88 
            # x[1] # BN, 256, 16, 44

        feat_img = x.unsqueeze(1)
        if self.training:
            x, depth = self.get_cam_feats(feat_img)
            batch_dict['depth_pred'] = depth
        else:
            x = self.get_cam_feats(feat_img)

        if self.accelerate and self.cache is not None:
            x = self.acc_bev_pool(x)
        else:
            lidar2image = batch_dict['sensor2image'].cuda()
            # print(lidar2image)
            
            geom = self.get_geometry(
                lidar2image
            )
            x = self.bev_pool(geom, x)
        
        x = self.downsample(x)
        batch_dict['cam_bev_feat'] = x.permute(0,1,3,2).contiguous()
        # print(x.shape)

        batch_dict = self.head(batch_dict)

        return batch_dict
    
    def loss(self, batch_dict):
        loss = 0.
        
        rpn_loss = self.head.loss(batch_dict)
        loss += rpn_loss

        ### depth loss ###
        depth = batch_dict['depth_pred']
        depth_label = batch_dict['depth_labels'].cuda()

        avail_mask = torch.where(depth_label > 0.)
        ind_b, ind_n, ind_h, ind_w = avail_mask
        d_min, d_max, d_interval = self.dbound
        depth_label_ind = torch.clamp((depth_label-d_min)/d_interval, 0, self.D-1).to(torch.int64)
        depth_label_ind_avail = depth_label_ind[avail_mask]
        depth_avail = depth[ind_b, ind_n, :, ind_h, ind_w] # N, D
        n_avail = depth_avail.shape[0]
        cross_entropy = -torch.sum(torch.log(depth_avail[torch.arange(0,n_avail),depth_label_ind_avail] + 1e-8)) / (n_avail + 1e-8)
        
        depth_loss = self.w_depth * cross_entropy
        
        # if self.clamp is not None:
        #     loss = torch.clamp(loss, max=self.clamp)

        if self.is_logging:
            batch_dict['logging'].update({
                'depth_loss': depth_loss.item(),
            })
        loss += depth_loss
        ### depth loss ###

        ### distillation loss ###
        if self.is_distil:
            if 'ldr_bev_feat' in batch_dict.keys():
                if self.mask_area == 'all':
                    if self.distil_loss_type == 'channel':
                        ldr_feat = batch_dict['ldr_bev_feat']
                        cam_feat = batch_dict['cam_bev_feat']
                    else:
                        ldr_feat = torch.mean(batch_dict['ldr_bev_feat'], dim=1)
                        cam_feat = torch.mean(batch_dict['cam_bev_feat'], dim=1)

                    distil_loss = self.w_distil * torch.mean((ldr_feat-cam_feat)**2)

                elif self.mask_area == 'fast_bev_pool':
                    ldr_feat = batch_dict['ldr_bev_feat']
                    cam_feat = batch_dict['cam_bev_feat']
                    
                    obj_mask = batch_dict['obj_mask'].cuda()
                    f_b, f_y, f_x = torch.where(obj_mask==True)   # Foreground
                    b_b, b_y, b_x = torch.where(obj_mask==False)  # Background

                    w_f, w_b = self.fb_rate

                    if self.distil_loss_type == 'channel':
                        foreground_loss = w_f*torch.mean((ldr_feat[f_b, :, f_y, f_x]-cam_feat[f_b, :, f_y, f_x])**2)
                        background_loss = w_b*torch.mean((ldr_feat[b_b, :, b_y, b_x]-cam_feat[b_b, :, b_y, b_x])**2)
                    else:
                        foreground_loss = w_f*((torch.mean(ldr_feat[f_b, :, f_y, f_x])-torch.mean(cam_feat[f_b, :, f_y, f_x]))**2)
                        background_loss = w_b*((torch.mean(ldr_feat[b_b, :, b_y, b_x])-torch.mean(cam_feat[b_b, :, b_y, b_x]))**2)

                    distil_loss = foreground_loss + background_loss

                    if self.is_logging:
                        batch_dict['logging'].update({
                            'foreground_loss': foreground_loss.item(),
                            'background_loss': background_loss.item(),
                        })
                    
                if self.is_logging:
                    batch_dict['logging'].update({
                        'distil_loss': distil_loss.item(),
                    })
                loss += distil_loss

                ### Vis ###
                # print(batch_dict['ldr_bev_feat'].shape)
                # print(batch_dict['cam_bev_feat'].shape)
                # import matplotlib.pyplot as plt
                # feat_vis = torch.mean(batch_dict['ldr_bev_feat'][0,:,:,:], dim=0)
                # plt.subplot(1,2,1)
                # plt.imshow(feat_vis.detach().cpu().numpy())
                # plt.colorbar()
                # plt.subplot(1,2,2)
                # plt.imshow(batch_dict['camera_imgs'][0,0,:,:,:].permute(1,2,0).detach().cpu())
                # plt.show()
                ### Vis ###
        ### distillation loss ###

        return loss
