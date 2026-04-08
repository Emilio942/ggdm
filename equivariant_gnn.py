import torch
import torch.nn as nn
import torch.nn.functional as F

class EGNN_Layer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network Layer.
    Erhält kontinuierliche Koordinaten (x) und diskrete Features (h).
    Aktualisiert beide unter strikter E(3)-Äquivarianz.
    """
    def __init__(self, hidden_dim, edge_dim=0, act_fn=nn.SiLU()):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Edge Message Network: phi_e
        # Input: h_i, h_j, squared distance, edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Coordinate Update Network: phi_x
        # Berechnet die skalare Kraft, die entlang des Vektors (x_i - x_j) wirkt
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1)
        )
        
        # Node Update Network: phi_h
        # Input: h_i, aggregierte Messages m_i
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        row, col = edge_index
        
        # 1. Berechne relative Vektoren und Abstände
        coord_diff = x[row] - x[col]                  # [num_edges, 3]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True) # [num_edges, 1]
        
        # 2. Berechne Messages für Kanten
        if edge_attr is not None:
            edge_input = torch.cat([h[row], h[col], radial, edge_attr], dim=1)
        else:
            edge_input = torch.cat([h[row], h[col], radial], dim=1)
            
        m_ij = self.edge_mlp(edge_input) # [num_edges, hidden_dim]
        
        # 3. Koordinaten-Update (Äquivariant!)
        # Die Kraft wirkt zentrisch entlang coord_diff
        force_scalar = self.coord_mlp(m_ij) # [num_edges, 1]
        force_vector = coord_diff * force_scalar # [num_edges, 3]
        
        # Aggregation der Kräfte auf die Knoten
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, force_vector)
        x_new = x + x_update
        
        # 4. Knoten-Feature-Update (Invariant!)
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, row, m_ij)
        
        node_input = torch.cat([h, m_i], dim=1)
        h_new = h + self.node_mlp(node_input)
        
        return h_new, x_new


class EquivariantScoreNetwork(nn.Module):
    """
    Das Haupt-Score-Netzwerk. Nimmt verrauschte Zustände (x_t, h_t) und t
    und sagt den Score voraus.
    """
    def __init__(self, num_atom_types, hidden_dim=128, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Atom-Typ Embedding (diskrete H in kontinuierliche Features umwandeln)
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # EGNN Layers
        self.layers = nn.ModuleList([
            EGNN_Layer(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output Heads
        # 1. Für den kategorialen Score (Übergangs-Logits für Atomtypen)
        self.h_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types)
        )
        
        # 2. Für den räumlichen Score (wird durch das EGNN intrinsisch als Vektor generiert)
        # Wir berechnen einen finalen force-Vektor
        self.final_coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_t, h_t, t, edge_index):
        """
        x_t: [N, 3] Koordinaten
        h_t: [N] Atom-Typ-Indizes
        t: [B, 1] Diffusion Time (für Batch)
        edge_index: [2, E] Adjazenzmatrix (z.B. fully connected für kleine Moleküle)
        """
        # Node Features initialisieren
        h = self.atom_embedding(h_t) # [N, hidden_dim]
        
        # Zeit-Embedding addieren (vereinfacht: gleiche Zeit für alle Knoten)
        t_emb = self.time_mlp(t)
        # Wir fügen t_emb zu den Knoten hinzu. In echtem PyG nimmt man batch.
        h = h + t_emb.expand_as(h) 
        
        # Message Passing durch EGNN
        # Wir behalten das initiale x, da wir nur den Gradienten (Score) vorhersagen wollen
        x_pred = x_t.clone()
        
        for layer in self.layers:
            h, x_pred = layer(h, x_pred, edge_index)
            
        # Output: Kategorien
        h_score = self.h_out(h) # [N, num_atom_types] Logits
        
        # Output: Koordinaten-Score (Vektorfeld)
        # Die Differenz zwischen x_pred und x_t ist unser E(3)-äquivarianter Score
        x_score = x_pred - x_t # [N, 3]
        
        return x_score, h_score
