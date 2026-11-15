import torch
import torch.nn as nn
from sam2.build_sam import build_sam2


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class SAM2(nn.Module):
    def __init__(self, checkpoint_path: str = None) -> None:
        super(SAM2, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        return {
            '0': x1,
            '1': x2,
            '2': x3,
            '3': x4
        }


if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out = model(x)
        print(out['0'].shape, out['1'].shape, out['2'].shape, out['3'].shape)