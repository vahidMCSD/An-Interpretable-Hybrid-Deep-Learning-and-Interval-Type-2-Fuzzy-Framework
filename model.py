import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )

    def forward(self, x):
        avg = x.mean(dim=(2,3))
        mx = x.amax(dim=(2,3))
        a = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).unsqueeze(-1).unsqueeze(-1)
        return x * a

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class EfficientNetCBAMBranch(nn.Module):
    def __init__(self, out_dim=128, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            features_only=True,
            in_chans=3
        )
        channels = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAM(channels, reduction=16)
        self.proj = nn.Linear(channels, out_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        x3 = x.repeat(1, 3, 1, 1)
        f = self.backbone(x3)[-1]
        f = self.cbam(f)
        f = f.mean(dim=(2,3))
        return self.proj(f)

class ViTBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        # ViT-B/16, ImageNet-21K style initialization where available in timm.
        # timm may map this to an available pretrained checkpoint depending on version.
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )
        dim = self.vit.num_features
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x):
        x3 = x.repeat(1, 3, 1, 1)
        z = self.vit(x3)
        return self.proj(z)

class AttentionFusion(nn.Module):
    def __init__(self, input_dim=404, hidden_dim=256, heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.proj(x)
        # one fused token, refined by MHSA exactly as described in the manuscript
        token = h.unsqueeze(1)
        a, _ = self.mhsa(token, token, token, need_weights=False)
        out = self.norm(token + a)
        return out.squeeze(1)

class HybridDeepModel(nn.Module):
    """
    Deep branches + attention fusion.
    Wavelet-PCA features are supplied externally as a 20-D tensor.
    """
    def __init__(self):
        super().__init__()
        self.eff = EfficientNetCBAMBranch(out_dim=128)
        self.vit = ViTBranch(out_dim=256)
        self.fusion = AttentionFusion(input_dim=20+128+256, hidden_dim=256, heads=4)

    def forward(self, image, wavelet20):
        e = self.eff(image)
        v = self.vit(image)
        x = torch.cat([wavelet20, e, v], dim=1)
        return self.fusion(x), e, v
