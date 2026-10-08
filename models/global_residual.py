"""Small absolute-location-conditioned residual for frozen LTE outputs.

LTE's native query coordinate remains crop-local. This adapter receives a
separate whole-domain coordinate and learns a bounded additive correction. Its
last layer starts at zero, so the initial output is exactly the base LTE.
"""

import math

import torch
import torch.nn as nn


class GlobalResidualAdapter(nn.Module):
    def __init__(self, fourier_bands=(1, 2, 4, 8), hidden_dim=32,
                 max_delta_norm=0.25, station_coords=None,
                 gate_radius_norm=0.15):
        super().__init__()
        bands = torch.as_tensor(fourier_bands, dtype=torch.float32)
        if bands.ndim != 1 or bands.numel() == 0 or (bands <= 0).any():
            raise ValueError('fourier_bands must be a non-empty list of positive values')
        if hidden_dim <= 0 or max_delta_norm <= 0:
            raise ValueError('hidden_dim and max_delta_norm must be positive')
        self.register_buffer('bands', bands)
        if station_coords is None:
            station_coords = torch.empty(0, 2)
        station_coords = torch.as_tensor(station_coords, dtype=torch.float32)
        if station_coords.ndim != 2 or station_coords.shape[-1] != 2:
            raise ValueError('station_coords must have shape [N, 2]')
        self.register_buffer('station_coords', station_coords)
        self.gate_radius_norm = float(gate_radius_norm)
        self.max_delta_norm = float(max_delta_norm)

        # base prediction + sin/cos for each (y, x) and frequency.
        in_dim = 1 + 4 * bands.numel()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _encode(self, global_coord):
        phase = 2.0 * math.pi * global_coord.unsqueeze(-1) * self.bands
        # [B, Q, 2, F] -> [B, Q, 4F]
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)

    def gate(self, global_coord):
        if self.station_coords.numel() == 0 or self.gate_radius_norm <= 0:
            return torch.ones_like(global_coord[..., :1])
        d2 = (global_coord.unsqueeze(-2) - self.station_coords).square().sum(dim=-1)
        return torch.exp(-d2 / (2.0 * self.gate_radius_norm ** 2)).amax(dim=-1, keepdim=True)

    def forward(self, base_pred, global_coord):
        if base_pred.shape[:-1] != global_coord.shape[:-1] or global_coord.shape[-1] != 2:
            raise ValueError('base_pred must be [B,Q,1] and global_coord [B,Q,2]')
        # The direct base_pred term receives its normal loss gradient. Detach
        # only the residual-head feature path for a better-conditioned update.
        x = torch.cat([base_pred.detach(), self._encode(global_coord)], dim=-1)
        delta = self.max_delta_norm * torch.tanh(self.net(x))
        return base_pred + self.gate(global_coord) * delta
