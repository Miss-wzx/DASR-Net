import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 1. 门控卷积 ====================
class GatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1, activation='elu'):
        super().__init__()
        self.conv_feat = nn.Conv2d(in_ch, out_ch, kernel, stride, pad)
        self.conv_gate = nn.Conv2d(in_ch, out_ch, kernel, stride, pad)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv_feat(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return self.act(feat * gate)

class GatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, upsample=False, activation='elu'):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = GatedConv2d(in_ch, out_ch, 3, 1, 1, activation)
        else:
            self.conv = GatedConv2d(in_ch, out_ch, 3, stride, 1, activation)

    def forward(self, x):
        if self.upsample:
            x = self.up(x)
        return self.conv(x)

# ==================== 2. 超图卷积层 ====================
class HypergraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_dim=64, n_hyperedges=32):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_dim = hidden_dim
        self.n_hyperedges = n_hyperedges

        self.psi_conv = nn.Conv2d(in_ch, hidden_dim, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lambda_fc = nn.Conv2d(in_ch, hidden_dim, 1)
        self.omega_conv = nn.Conv2d(in_ch, n_hyperedges, 7, padding=3)

        self.theta = nn.Parameter(torch.Tensor(hidden_dim, out_ch))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        psi = self.psi_conv(x).view(B, self.hidden_dim, N).permute(0, 2, 1)          # B,N,hidden
        lam = self.gap(x)
        lam = self.lambda_fc(lam).view(B, self.hidden_dim)                           # B,hidden
        Lambda = torch.diag_embed(lam)                                              # B,hidden,hidden

        omega = self.omega_conv(x).view(B, self.n_hyperedges, N).permute(0, 2, 1)   # B,N,M
        omega = torch.abs(omega) + 1e-8

        psiL = torch.matmul(psi, Lambda)                                            # B,N,hidden
        psiL_psiT = torch.matmul(psiL, psi.transpose(1, 2))                         # B,N,N
        H_assoc = torch.matmul(psiL_psiT, omega)                                    # B,N,M
        H_assoc = torch.abs(H_assoc) + 1e-8

        H_T = H_assoc.permute(0, 2, 1)                                              # B,M,N
        D_diag = H_assoc.sum(dim=2) + 1e-8                                          # B,N
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(D_diag))                    # B,N,N
        B_diag = H_assoc.sum(dim=1) + 1e-8                                          # B,M
        B_inv = torch.diag_embed(1.0 / B_diag)                                      # B,M,M

        temp = torch.matmul(H_assoc, B_inv)                                         # B,N,M
        temp = torch.matmul(temp, H_T)                                              # B,N,N
        L = torch.matmul(D_inv_sqrt, temp)
        L = torch.matmul(L, D_inv_sqrt)
        eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        Delta = eye - L

        X_flat = psi                                                                # B,N,hidden
        out = torch.matmul(Delta, X_flat)                                           # B,N,hidden
        out = torch.matmul(out, self.theta)                                         # B,N,out_ch
        out = out.permute(0, 2, 1).view(B, self.out_ch, H, W)
        out = F.elu(out)
        return out

# ==================== 3. Coarse 生成器（保持分辨率） ====================
class CoarseGenerator(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.down1 = GatedConvBlock(in_ch, 64, stride=2)
        self.down2 = GatedConvBlock(64, 128, stride=2)
        self.down3 = GatedConvBlock(128, 256, stride=2)
        self.down4 = GatedConvBlock(256, 512, stride=2)

        self.bottleneck = GatedConv2d(512, 512)

        self.up4 = GatedConvBlock(512, 256, upsample=True)
        self.up3 = GatedConvBlock(512, 128, upsample=True)   # 跳跃连接后通道翻倍
        self.up2 = GatedConvBlock(256, 64, upsample=True)
        self.up1 = GatedConvBlock(128, in_ch, upsample=True)

    def forward(self, x):
        d1 = self.down1(x)   # 16
        d2 = self.down2(d1)  # 8
        d3 = self.down3(d2)  # 4
        d4 = self.down4(d3)  # 2

        b = self.bottleneck(d4)  # 2

        u4 = self.up4(b)                     # 4
        u4 = torch.cat([u4, d3], dim=1)      # 256+256=512
        u3 = self.up3(u4)                    # 8
        u3 = torch.cat([u3, d2], dim=1)      # 128+128=256
        u2 = self.up2(u3)                    # 16
        u2 = torch.cat([u2, d1], dim=1)      # 64+64=128
        out = self.up1(u2)                   # 32
        return out

# ==================== 4. Refine 生成器（超图卷积 + 上采样至 80x80） ====================
class RefineGenerator(nn.Module):
    def __init__(self, in_ch=12, out_ch=6, target_size=(80,80)):
        super().__init__()
        self.target_h, self.target_w = target_size

        self.down1 = GatedConvBlock(in_ch, 64, stride=2)
        self.down2 = GatedConvBlock(64, 128, stride=2)
        self.down3 = GatedConvBlock(128, 256, stride=2)
        self.down4 = GatedConvBlock(256, 512, stride=2)

        self.hypergraph = HypergraphConv(512, 512, hidden_dim=64, n_hyperedges=32)

        self.up4 = GatedConvBlock(512, 256, upsample=True)
        self.up3 = GatedConvBlock(512, 128, upsample=True)
        self.up2 = GatedConvBlock(256, 64, upsample=True)

        self.final = nn.Sequential(
            nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
            GatedConv2d(128, 64, 3, 1, 1),
            GatedConv2d(64, out_ch, 3, 1, 1)
        )

    def forward(self, x, coarse_out=None):
        if coarse_out is not None:
            x = torch.cat([x, coarse_out], dim=1)   # 输入 6+6=12 通道

        d1 = self.down1(x)   # 16
        d2 = self.down2(d1)  # 8
        d3 = self.down3(d2)  # 4
        d4 = self.down4(d3)  # 2

        hg = self.hypergraph(d4)   # 超图卷积增强全局特征

        u4 = self.up4(hg)          # 4
        u4 = torch.cat([u4, d3], dim=1)
        u3 = self.up3(u4)          # 8
        u3 = torch.cat([u3, d2], dim=1)
        u2 = self.up2(u3)          # 16
        u2 = torch.cat([u2, d1], dim=1)

        out = self.final(u2)       # 80x80
        return out

# ==================== 5. 完整模型 ====================
class HypergraphInpainting(nn.Module):
    def __init__(self, in_ch=6, target_size=(80,80)):
        super().__init__()
        self.coarse = CoarseGenerator(in_ch)
        self.refine = RefineGenerator(in_ch*2, in_ch, target_size)

    def forward(self, x):
        coarse_out = self.coarse(x)          # (B,6,32,32)
        refine_out = self.refine(x, coarse_out)  # (B,6,80,80)
        return refine_out
        # return refine_out, coarse_out

# ==================== 6. 简单测试 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HypergraphInpainting(in_ch=6, target_size=(80,80)).to(device)
    dummy = torch.randn(2, 6, 32, 32).to(device)
    with torch.no_grad():
        final, coarse = model(dummy)
    print("Input shape :", dummy.shape)     # [2,6,32,32]
    print("Coarse shape:", coarse.shape)    # [2,6,32,32]
    print("Final shape :", final.shape)     # [2,6,80,80]

