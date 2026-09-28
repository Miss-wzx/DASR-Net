# ============================================================
# Depth-Aware Super-Resolution Reconstruction Network for Scanning Acoustic Microscopy (DASR-Net)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Residual Block
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels*4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels*4, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.block(x)


# ============================================================
# Global CNN Encoder Input: [B, 6, 32, 32] Output: [B, dim, 32, 32]
# ============================================================

class GlobalEncoder(nn.Module):

    def __init__(self, in_channels=6, dim=64):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU()
        )

        self.body = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        )

    def forward(self, x):
        """
        x: [B, 6, 32, 32]
        """
        x = self.head(x)
        x = self.body(x)
        return x


# ============================================================
# Cross Depth Attention
# ============================================================

class CrossDepthAttention(nn.Module):
    def __init__(self, channels):

        super().__init__()

        self.query = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.key = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.value = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            ResBlock(channels),

            ResBlock(channels)
        )

    def forward(self, x):

        """
        x:
        [B,D,C,H,W]
        """

        B, D, C, H, W = x.shape

        outputs = []

        for i in range(D):

            current = x[:, i]

            q = self.query(current)

            depth_context = 0

            count = 0

            # ====================================================
            # Previous slice interaction
            # ====================================================

            if i > 0:
                prev_feat = x[:, i - 1]

                k_prev = self.key(prev_feat)

                v_prev = self.value(prev_feat)

                attn_prev = torch.sigmoid(q * k_prev)

                depth_context += attn_prev * v_prev

                count += 1

            # ====================================================
            # Next slice interaction
            # ====================================================

            if i < D - 1:
                next_feat = x[:, i + 1]

                k_next = self.key(next_feat)

                v_next = self.value(next_feat)

                attn_next = torch.sigmoid(q * k_next)

                depth_context += attn_next * v_next

                count += 1

            if count > 0:
                depth_context = depth_context / count

            fused = self.fusion(

                torch.cat(
                    [current, depth_context],
                    dim=1
                )
            )

            outputs.append(fused)

        out = torch.stack(outputs, dim=1)

        return out





# ============================================================
# Depth Aware Encoder Input: [B, 6, 32, 32] Output: [B, dim, 32, 32]
# ============================================================

class DepthAwareEncoder(nn.Module):

    def __init__(self, dim=64):
        super().__init__()

        self.embed_list = nn.ModuleList([
            nn.Conv2d(1, dim, 3, 1, 1)
            for _ in range(6)
        ])

        # self.embed = nn.Conv2d(1, dim, 3, 1, 1)

        self.res_block  = ResBlock(dim)

        # Cross Depth Attention ×2
        self.cda1 = CrossDepthAttention(dim)

        self.cda2 = CrossDepthAttention(dim)


    def forward(self, x):
        """
        x: [B, D, H, W]
        """

        B, D, H, W = x.shape

        features = []

        for i in range(D):
            feat = self.embed_list[i](
                x[:, i:i + 1]
            )

        # for i in range(D):
        #
        #     feat = self.embed(x[:, i:i+1])

            feat = self.res_block(feat)

            features.append(feat)

        depth_features  = torch.stack(features, dim=1)

        # ==========================================
        # Cross Depth Attention ×2
        # ==========================================

        depth_features = self.cda1(depth_features)

        depth_features = self.cda2(depth_features)

        # ==========================================
        # Mean Pooling over depth
        # ==========================================

        spatial_feature = depth_features.mean(dim=1)

        return spatial_feature


# ============================================================
# Reconstruct Decoder
# ============================================================
class ReconstructDecoder(nn.Module):

    def __init__(self, dim=64, out_frames=6):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
            nn.GELU(),
            ResBlock(dim),
            ResBlock(dim),
        )

        # 32 -> 64
        self.up1 = nn.ConvTranspose2d(
            dim,
            dim,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # 64 -> 80
        self.up2 = nn.Sequential(

            nn.Upsample(
                size=(80, 80),
                mode='bilinear',
                align_corners=False
            ),

            nn.Conv2d(
                dim,
                dim,
                3,
                1,
                1
            )
        )

        self.reconstruct = nn.Sequential(
            nn.GELU(),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.Conv2d(dim, out_frames, 3, 1, 1)
        )

        self.output_act = nn.Tanh()

    def forward(self, global_feat, depth_feat):

        feat = torch.cat(
            [global_feat, depth_feat],
            dim=1
        )

        feat = self.fusion(feat)

        feat = self.up1(feat)

        feat = self.up2(feat)

        out = self.reconstruct(feat)

        out = self.output_act(out)

        return out

# ============================================================
# DASR-Net
# ============================================================

class DASRNet(nn.Module):

    def __init__(self,
                 in_frames=6,
                 dim=64):

        super().__init__()

        self.global_encoder = GlobalEncoder(
            in_channels=in_frames,
            dim=dim
        )

        self.depth_aware_encoder = DepthAwareEncoder(
            dim=dim
        )

        self.fusion_and_reconstruct_decoder = ReconstructDecoder(dim)

    def forward(self, x):
        """
        x:
        [B, D, H, W]

        """

        global_feat = self.global_encoder(x)

        depth_feat = self.depth_aware_encoder(x)

        out = self.fusion_and_reconstruct_decoder(global_feat, depth_feat)

        return out



# ============================================================
# Loss Functions
# ============================================================
# ============================================================
# Gradient Loss
# ============================================================

class GradientLoss(nn.Module):

    def __init__(self):

        super().__init__()

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float()

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float()

        self.register_buffer(
            'sobel_x',
            sobel_x.unsqueeze(0).unsqueeze(0)
        )

        self.register_buffer(
            'sobel_y',
            sobel_y.unsqueeze(0).unsqueeze(0)
        )

    def gradient(self, x):

        x = F.pad(
            x,
            (1, 1, 1, 1),
            mode='reflect'
        )

        gx = F.conv2d(
            x,
            self.sobel_x
        )

        gy = F.conv2d(
            x,
            self.sobel_y
        )

        return gx, gy

    def forward(self, pred, gt):

        B, D, H, W = pred.shape

        pred = pred.reshape(
            B * D,
            1,
            H,
            W
        )

        gt = gt.reshape(
            B * D,
            1,
            H,
            W
        )

        pred_gx, pred_gy = self.gradient(pred)
        gt_gx, gt_gy = self.gradient(gt)

        loss_x = F.l1_loss(
            pred_gx,
            gt_gx,
            reduction='none'
        )

        loss_y = F.l1_loss(
            pred_gy,
            gt_gy,
            reduction='none'
        )

        loss_x = loss_x.reshape(
            B,
            D,
            H,
            W
        )

        loss_y = loss_y.reshape(
            B,
            D,
            H,
            W
        )

        loss = (
            loss_x.mean(dim=(1, 2, 3))
            +
            loss_y.mean(dim=(1, 2, 3))
        ) * 0.5

        return loss


# ============================================================
# Depth Consistency Loss
# ============================================================

class DepthConsistencyLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, pred, gt):

        pred_diff = pred[:, 1:] - pred[:, :-1]

        gt_diff = gt[:, 1:] - gt[:, :-1]

        loss = F.l1_loss(
            pred_diff,
            gt_diff,
            reduction='none'
        )

        loss = loss.mean(
            dim=(1, 2, 3)
        )

        return loss

# 拉普拉斯
class LaplacianLoss(nn.Module):

    def __init__(self):

        super().__init__()

        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]).float()

        self.register_buffer(
            "kernel",
            kernel.unsqueeze(0).unsqueeze(0)
        )

    def forward(self, pred, gt):

        B, D, H, W = pred.shape

        pred = pred.reshape(
            B * D,
            1,
            H,
            W
        )

        gt = gt.reshape(
            B * D,
            1,
            H,
            W
        )

        pred_edge = F.conv2d(
            F.pad(
                pred,
                (1, 1, 1, 1),
                mode='reflect'
            ),
            self.kernel
        )

        gt_edge = F.conv2d(
            F.pad(
                gt,
                (1, 1, 1, 1),
                mode='reflect'
            ),
            self.kernel
        )

        loss = F.l1_loss(
            pred_edge,
            gt_edge,
            reduction='none'
        )

        loss = loss.reshape(
            B,
            D,
            H,
            W
        )

        loss = loss.mean(
            dim=(1, 2, 3)
        )

        return loss

# ============================================================
# Total Loss
# ============================================================

class TotalLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.grad = GradientLoss()

        self.depth = DepthConsistencyLoss()

        self.lap = LaplacianLoss()

    def forward(self, pred, gt):

        # ----------------------------------------------------
        # L1
        # ----------------------------------------------------

        l1_loss = F.l1_loss(
            pred,
            gt,
            reduction='none'
        )

        l1_loss = l1_loss.mean(
            dim=(1, 2, 3)
        )

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        grad_loss = self.grad(
            pred,
            gt
        )

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        depth_loss = self.depth(
            pred,
            gt
        )

        # --------------------------
        # Edge
        # --------------------------

        lap_loss = self.lap(
            pred,
            gt
        )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        total_loss = (
            0.2 * l1_loss # 均质化
            +
            0.4 * grad_loss
            +
            0.1 * depth_loss  # 均质化
            +
            0.4 * lap_loss
        )

        return {

            'total': total_loss,      # [B]

            'l1': l1_loss,            # [B]

            'grad': grad_loss,        # [B]

            'depth': depth_loss,       # [B]

            'lap': lap_loss
        }

# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ========================================================
    # Create model
    # ========================================================

    model = DASRNet(
        in_frames=6,
        dim=64
    ).to(device)

    loss = TotalLoss().to(device)


    # ========================================================
    # Print model
    # ========================================================

    print(model)
    print('---------------------------------')

    # ========================================================
    # Random input
    # Shape:
    # [B,6,32,32]
    # ========================================================

    x = torch.randn(
        4,      # batch size
        6,
        32,
        32
    ).to(device)

    y = torch.randn(
        4,      # batch size
        6,
        80,
        80
    ).to(device)

    # ========================================================
    # Forward
    # ========================================================

    y_hat = model(x)
    print('---------------------------------')

    l = loss(y_hat, y)

    print('loss:', l)
    print('loss:', l['total'].shape)


    print("\nInput shape :", x.shape)
    print("Output shape:", y_hat.shape)

    # ========================================================
    # Parameter count
    # ========================================================

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # ========================================================
    # Output statistics
    # ========================================================

    print("\nOutput statistics:")
    print("Mean :", y_hat.mean().item())
    print("Std  :", y_hat.std().item())
    print("Min  :", y_hat.min().item())
    print("Max  :", y_hat.max().item())