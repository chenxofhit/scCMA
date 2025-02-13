import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse
from loss_contrastive import SupConLoss

class scCMA(torch.nn.Module):
    def __init__(
            self,
            num_genes,
            hidden_size=128,
            dropout=0,
            masked_data_weight=.75,
            mask_loss_weight=0.7,
            recon_loss_weight=0.23,
            contrastive_weight=0.07,
            temperature=0.07,
            proj_dim=128,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.contrastive_weight = contrastive_weight
        self.supcon_loss = SupConLoss(temperature=temperature)

        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_genes, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, proj_dim)
        )

        self.mask_predictor = nn.Linear(hidden_size, num_genes)
        self.decoder = nn.Linear(
            in_features=hidden_size + num_genes, out_features=num_genes)

    def forward_mask(self, x):
        latent = self.encoder(x)
        predicted_mask = self.mask_predictor(latent)
        reconstruction = self.decoder(
            torch.cat([latent, predicted_mask], dim=1))

        return latent, predicted_mask, reconstruction

    def loss_mask(self, x, y, mask, labels=None):
        latent, predicted_mask, reconstruction = self.forward_mask(x)
        w_nums = mask * self.masked_data_weight + (1 - mask) * (1 - self.masked_data_weight)
        reconstruction_loss = (1 - self.mask_loss_weight - self.contrastive_weight) * torch.mul(
            w_nums, mse(reconstruction, y, reduction='none'))

        mask_loss = self.mask_loss_weight * \
                    bce_logits(predicted_mask, mask, reduction="mean")
        reconstruction_loss = reconstruction_loss.mean()

        if labels is not None:

            bsz = x.shape[0] // 2
            latent1, latent2 = latent[:bsz], latent[bsz:]

            proj1 = self.projection_head(latent1)
            proj2 = self.projection_head(latent2)

            proj1 = F.normalize(proj1, dim=-1)
            proj2 = F.normalize(proj2, dim=-1)

            features = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)

            contrastive_loss = self.supcon_loss(features, labels[:bsz])
            total_loss = reconstruction_loss + mask_loss + self.contrastive_weight * contrastive_loss
        else:
            total_loss = reconstruction_loss + mask_loss
        return latent, total_loss

    def feature(self, x):
        latent = self.encoder(x)
        return latent
