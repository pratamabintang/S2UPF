import torch
from torch import nn
import torch.nn.functional as F
from .loss import WeightedDiceLoss
from .transformer import PositionEmbeddingSine
from .transformer import Transformer
from .encoder import SAM2
from .decoder import Decoder

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

class ProtoFormer(nn.Module):
    def __init__(
        self,
        shot: int = 1,
        reduce_dim: int = 64,
        img_size: int = 512,
        use_prototype: bool = True,
        use_prior: bool = True,
        criterion = None,
        checkpoint_path: str = None,
        model_cfg: str = "sam2_hiera_l.yaml"
    ):
        super(ProtoFormer, self).__init__()
        if img_size % 32 != 0:
            raise ValueError('img_size must be a multiple of 32, got %d' % img_size)
        self.criterion = criterion if criterion is not None else WeightedDiceLoss()
        self.shot = shot
        self.reduce_dim = reduce_dim

        self.backbone = SAM2(checkpoint_path=checkpoint_path, model_cfg=model_cfg)
        channels = list(reversed(self.backbone.channel_list))
        in_fea_dim = channels[1] + channels[2]
        self.high_dim = channels[3]

        drop_out = 0.5
        self.adjust_feature_supp = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        ) 
        self.adjust_feature_qry = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        )
        self.query_embed=nn.Embedding(reduce_dim,1) 
        self.pe_layer=PositionEmbeddingSine(reduce_dim//2, normalize=True)
        self.transformer = Transformer(
            d_model=reduce_dim,
            dropout=0.1,
            nhead=4,
            dim_feedforward=reduce_dim//4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        self.high_avg_pool = nn.Identity()        
        self.img_size = img_size
        self.use_prototype = use_prototype
        self.use_prior = use_prior
        self.decoder = Decoder(
            x1_ch=channels[0],
            x2_ch=channels[1],
            x3_ch=channels[2],
            x4_ch=channels[3],
            proto_dim=reduce_dim,
            use_prototype=use_prototype,
            use_prior=use_prior
        )
        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if name == 'backbone' or name.startswith('backbone.'):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(
        self,
        x,
        s_x=None,
        s_y=None
    ):
        if s_x is None or s_y is None:
            raise ValueError("support missing")

        batch_size, _, height, width = x.size()
        img_size = x.size()[-2:]

        # backbone
        all_imgs = torch.cat([x, s_x.view(-1, 3, *img_size)], dim=0)
        all_fts = self.backbone(all_imgs)

        # separate
        qry_bcb_fts = {}
        supp_bcb_fts = {}
        for key in all_fts.keys():
            qry_bcb_fts[key] = all_fts[key][:batch_size]
            supp_bcb_fts[key] = all_fts[key][batch_size:]


        query_feat_2 = F.interpolate(
            qry_bcb_fts['2'],
            size=qry_bcb_fts['1'].shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        supp_feat_2 = F.interpolate(
            supp_bcb_fts['2'],
            size=supp_bcb_fts['1'].shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        query_feat = torch.cat([qry_bcb_fts['1'], query_feat_2], dim=1)
        supp_feat = torch.cat([supp_bcb_fts['1'], supp_feat_2], dim=1)

        query_feat = self.adjust_feature_qry(query_feat)
        supp_feat = self.adjust_feature_supp(supp_feat) 

        fts_size = query_feat.shape[-2:]
        supp_feat_list = []
        r_supp_feat = supp_feat.view(batch_size, self.shot, -1, fts_size[0], fts_size[1])
        for st in range(self.shot):
            mask = (s_y[:,st,:,:]==1).float().unsqueeze(1)
            mask = F.interpolate(mask, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=False)
            tmp_supp_feat = r_supp_feat[:,st,...]
            tmp_supp_feat = Weighted_GAP(tmp_supp_feat, mask)
            supp_feat_list.append(tmp_supp_feat)
        global_supp_pp = supp_feat_list[0]
        if self.shot > 1:
            for i in range(1, len(supp_feat_list)):
                global_supp_pp += supp_feat_list[i]
            global_supp_pp /= len(supp_feat_list)
                        
        query_embed = global_supp_pp.squeeze(-1)
        query_pos = self.query_embed.weight
        key_pos=self.pe_layer(query_feat)
        key_embed=query_feat
        masking=None
        if self.use_prototype:
            fg_embed=self.transformer(key_embed,masking,query_embed,query_pos,key_pos)
            prototype = fg_embed.squeeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            prototype = None
        # prior generation
        query_feat_high = qry_bcb_fts['3']
        high_size = supp_bcb_fts['3'].shape[-2:]
        supp_feat_high = supp_bcb_fts['3'].view(
            batch_size, self.shot, self.high_dim, high_size[0], high_size[1]
        )
        if self.use_prior:
            corr_query_mask = self.generate_prior(query_feat_high, supp_feat_high, s_y, fts_size)
        else:
            corr_query_mask = None

        logits = self.decoder(qry_bcb_fts, prototype, corr_query_mask)
        logits = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)

    def predict_mask(self, batch):
        logit_mask = self(batch['query_img'], batch['support_imgs'], batch['support_masks'])
        org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=False)
        return logit_mask
    
    def compute_loss(self, logit_mask, gt_mask):
        return self.criterion(logit_mask, gt_mask.long())

    def generate_prior(self, query_feat_high, supp_feat_high, s_y, fts_size):
        bsize, _, qh, qw = query_feat_high.size()
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for st in range(self.shot):
            tmp_mask = (s_y[:,st,:,:] == 1).float().unsqueeze(1)
            tmp_mask = F.interpolate(tmp_mask, size=supp_feat_high.shape[-2:], mode='bilinear', align_corners=False)

            tmp_supp_feat = supp_feat_high[:,st,...] * tmp_mask               
            q = self.high_avg_pool(query_feat_high.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            s = self.high_avg_pool(tmp_supp_feat.flatten(2).transpose(-2, -1))  # [bs, h*w, c]

            tmp_query = q.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, qh, qw)   
            similarity = (similarity.flatten(1) - similarity.flatten(1).min(1)[0].unsqueeze(1)) / (similarity.flatten(1).max(1)[0].unsqueeze(1) - similarity.flatten(1).min(1)[0].unsqueeze(1) + cosine_eps)
            similarity = similarity.view(bsize, 1, qh, qw)
            corr_query = F.interpolate(similarity, size=fts_size, mode='bilinear', align_corners=False)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        return corr_query_mask
