import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE3D(nn.Module):
    """
    Your 2D VAE used Linear layers (Flatten → Linear → Linear).
    That worked for small 2D images, but 64³ = 262,144 voxels.
    A Linear(262144, 784) layer alone = 205 MILLION parameters.
    
    Solution: Conv3d layers shrink spatial dims while learning local patterns.
    Only flatten at the very end when the volume is tiny (4³ = 64 values per channel).
    
    2D → 3D changes:
        Conv2d(in, out, k, s, p)  →  Conv3d(in, out, k, s, p)   # kernel is now a cube
        (B, C, H, W)             →  (B, C, D, H, W)             # extra depth axis
        bilinear upsampling       →  trilinear upsampling         # interpolate in 3 axes
        BatchNorm                 →  GroupNorm                    # doesn't need large batches
    """

    def __init__(self, z_dim=16, in_channels=1, input_size=64):
        super().__init__()
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.input_size = input_size

        # 4 stride-2 convs: 64 → 32 → 16 → 8 → 4
        self.bottleneck_spatial = input_size // 16   # = 4

        # ---- Encoder ----
        # Each block halves spatial dimensions, doubles channels
        self.encoder = nn.Sequential(
            # (B, 1, 64, 64, 64) → (B, 32, 32, 32, 32)
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),       # 8 groups × 4 channels each
            nn.LeakyReLU(0.2),         # LeakyReLU > ReLU: no dead neurons in encoder

            # (B, 32, 32³) → (B, 64, 16³)
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),

            # (B, 64, 16³) → (B, 128, 8³)
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),

            # (B, 128, 8³) → (B, 256, 4³)
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2),
        )

        # NOW we flatten: 256 channels × 4³ spatial = 16,384
        # Compare to your 2D VAE flattening at the start: 784 * height * width
        flat_size = 256 * (self.bottleneck_spatial ** 3)

        # Same mu/log_var split as your 2D VAE
        self.fc_mu = nn.Linear(flat_size, z_dim)
        self.fc_log_var = nn.Linear(flat_size, z_dim)

        # ---- Decoder ----
        self.fc_decode = nn.Linear(z_dim, flat_size)

        # Mirror of encoder, but using Upsample + Conv instead of ConvTranspose3d
        # Why? ConvTranspose creates checkerboard artifacts. Upsample + Conv is cleaner.
        self.decoder = nn.Sequential(
            # (B, 256, 4³) → (B, 128, 8³)
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),

            # (B, 128, 8³) → (B, 64, 16³)
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),

            # (B, 64, 16³) → (B, 32, 32³)
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),

            # (B, 32, 32³) → (B, 1, 64³)
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(32, in_channels, kernel_size=3, stride=1, padding=1),
            # NO activation here!
            # Your 2D VAE used Sigmoid because pixel values were [0, 1].
            # MRI crops are z-scored: mean=0, std=1, values can be negative.
            # So no sigmoid, and MSE loss instead of BCE.
        )

    def encode(self, x):
        h = self.encoder(x)          # (B, 256, 4, 4, 4)
        h = h.flatten(start_dim=1)   # (B, 16384)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        # Identical to your 2D version
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.fc_decode(z)        # (B, 16384)
        s = self.bottleneck_spatial
        h = h.view(-1, 256, s, s, s) # (B, 256, 4, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z  # ← return z too, needed for β-TCVAE

def beta_tcvae_loss(x, x_recon, mu, log_var, z, beta=4.0, dataset_size=458):
    batch_size = x.shape[0]   # M
    d = z.shape[1]             # latent dimensions

    # ================================================================
    # STEP 1: Reconstruction loss
    # ================================================================
    # MSE, not BCE — your data is z-scored, not [0,1] probabilities
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size

    # ================================================================
    # STEP 2: Build the (M × M × d) log-probability tensor
    # ================================================================
    # Goal: for every pair (i, j) in the batch, compute
    #   log q(z_i | x_j) = log N(z_i ; μ_j, diag(exp(logvar_j)))
    #
    # Why every pair? Because TC requires estimating q(z),
    # which is the average of q(z|x) over ALL x in the dataset.
    # We approximate with the minibatch.

    z_expand  = z.unsqueeze(1)           # (M, 1, d)  — each z_i
    mu_expand = mu.unsqueeze(0)          # (1, M, d)  — each μ_j
    lv_expand = log_var.unsqueeze(0)     # (1, M, d)  — each logvar_j

    # log N(z; μ, σ²) = -0.5 * [ log(2π) + logvar + (z-μ)²/exp(logvar) ]
    # Computed per-dimension: (M, M, d)
    log_qz_per_dim = -0.5 * (
        math.log(2 * math.pi)
        + lv_expand
        + (z_expand - mu_expand) ** 2 / torch.exp(lv_expand)
    )

    # Sum over d dimensions to get joint: (M, M)
    # Entry [i, j] = log q(z_i | x_j)  (all d dimensions jointly)
    log_qz_x_matrix = log_qz_per_dim.sum(dim=2)

    # ================================================================
    # STEP 3: Extract the three terms
    # ================================================================

    # --- Term A: log q(z_i | x_i) ---
    # Diagonal of the matrix: encoder evaluating its OWN sample
    log_qz_given_xi = log_qz_x_matrix.diag()           # (M,)

    # --- Term B: log q(z_i) ---
    # q(z) = (1/N) Σ_j q(z|x_j)
    # log q(z) ≈ logsumexp over j in batch - log(N)
    log_qz = (
        torch.logsumexp(log_qz_x_matrix, dim=1)        # (M,)
        - math.log(dataset_size)
    )

    # --- Term C: log ∏_k q(z_{i,k}) ---
    # Same idea but per dimension, THEN sum
    # If z dims were independent, B and C would be equal.
    # The gap between them IS the Total Correlation.
    log_qzk = (
        torch.logsumexp(log_qz_per_dim, dim=1)         # (M, d)
        - math.log(dataset_size)
    )
    log_prod_qzk = log_qzk.sum(dim=1)                  # (M,)

    # --- Term D: log p(z) ---
    # Prior: N(0, I)
    log_pz = -0.5 * (
        d * math.log(2 * math.pi) + (z ** 2).sum(dim=1)
    )                                                    # (M,)

    # ================================================================
    # STEP 4: The decomposition
    # ================================================================
    mi         = (log_qz_given_xi - log_qz).mean()      # A - B
    tc         = (log_qz - log_prod_qzk).mean()          # B - C
    dimwise_kl = (log_prod_qzk - log_pz).mean()          # C - D

    # Verify: mi + tc + dimwise_kl ≈ standard KL
    # (log_qz_given_xi - log_pz).mean() ≈ mi + tc + dimwise_kl ✓

    # ================================================================
    # STEP 5: Only upweight TC
    # ================================================================
    total = recon_loss + mi + beta * tc + dimwise_kl

    return total, {
        'recon': recon_loss.item(),
        'mi': mi.item(),
        'tc': tc.item(),
        'dimwise_kl': dimwise_kl.item(),
    }

@torch.no_grad()
def latent_traversal(model, x_sample, device, delta=1.0):
    """
    The visualization you described:
    Encode x → z, change one z_i at a time, decode, see what changes.

    Returns: diff_maps (z_dim, D, H, W) — how much each dim affects each voxel
    """
    model.eval()
    x = x_sample.unsqueeze(0).to(device)
    mu, _ = model.encode(x)
    z_base = mu[0]
    x_base = model.decode(z_base.unsqueeze(0))[0, 0]

    diff_maps = []
    for dim in range(model.z_dim):
        z_mod = z_base.clone()
        z_mod[dim] += delta
        x_mod = model.decode(z_mod.unsqueeze(0))[0, 0]
        diff_maps.append((x_mod - x_base).abs().cpu())

    return torch.stack(diff_maps)