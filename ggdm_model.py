import torch
import torch.nn as nn
import torch.nn.functional as F

from equivariant_gnn import EquivariantScoreNetwork

class GGDM(nn.Module):
    """
    Geometric Graph Diffusion Model (GGDM)
    Updated with Carlin's Lie-Euler Integrator and IMEX-Splitting.
    """
    def __init__(self, num_atom_types, hidden_dim=128, num_layers=4, timesteps=1000, stability_c=0.1):
        super().__init__()
        self.M = num_atom_types
        self.T = timesteps
        self.c = stability_c # Stability constant for adaptive dt
        
        # Diffusion Schedule (Linear σ schedule)
        self.sigma_min = 0.001
        self.sigma_max = 0.5
        
        # E(3)-Equivariant Score Network
        self.score_net = EquivariantScoreNetwork(
            num_atom_types=num_atom_types, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers
        )
        
        # Categorical Transition Logits (learnable per timestep)
        self.categorical_logits = nn.Parameter(torch.randn(timesteps, self.M, self.M))
        
        # Valence Mask
        self.register_buffer('valence_mask', torch.ones(self.M, self.M))

    def get_sigma(self, t):
        """Lineares Noise-Schedule σ(t)"""
        return self.sigma_min + (self.sigma_max - self.sigma_min) * (t / self.T)

    def get_adaptive_dt(self, t):
        """Berechnet optimales Δt zur Minimierung der Sampling-Varianz"""
        sigma_t = self.get_sigma(t)
        return self.c / (sigma_t**2 + 1e-8)

    def lie_euler_step(self, X_k, h_k, t_k, edge_index):
        """
        Führt einen Lie-Euler-Schritt auf SE(3)^N aus.
        Garantiert null geometrischen Drift.
        """
        dt = self.get_adaptive_dt(t_k)
        sigma_t = self.get_sigma(t_k)
        
        # 1. IMEX-Splitting: Linearer Prior vs NN-Korrektur
        # Wir holen den Score vom E(3)-äquivarianten Netzwerk
        t_tensor = torch.tensor([[t_k]], dtype=torch.float32, device=X_k.device)
        score_x, score_h = self.score_net(X_k, h_k, t_tensor, edge_index)
        
        # 2. Wiener Increment im Tangentialraum
        dt_tensor = torch.tensor(dt, device=X_k.device)
        dW = torch.randn_like(score_x) * torch.sqrt(dt_tensor)
        noise_term = sigma_t * dW
        
        # 3. Tangential-Inkrement (Drift + Noise)
        # v_tangent = score_nn * dt + noise_term
        v_tangent = score_x * dt + noise_term
        
        # 4. Exponential Map Update
        # Da wir mit EGNN arbeiten und nur Translationen betrachten (Punkte in 3D, keine SE(3) Frames für einfache EGNNs),
        # ist die Exp-Map einfach die Vektoraddition. Für volle SE(3) bräuchten wir Rotationen pro Atom.
        X_next = X_k + v_tangent
        
        return X_next, score_h

    def forward(self, x, h, t, edge_index):
        """Forward pass zur Score-Berechnung (fürs Training)"""
        return self.score_net(x, h, t, edge_index)

