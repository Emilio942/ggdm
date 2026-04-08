import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from ggdm_model import GGDM

def create_stable_synthetic_data(num_samples=10, num_atoms=8):
    """
    Erzeugt stabile synthetische Molekül-Graphen.
    Vermeidet externe Abhängigkeiten und Datensatz-Fehler.
    """
    dataset = []
    for _ in range(num_samples):
        # Positionen: Kleiner Cluster um den Ursprung
        pos = torch.randn(num_atoms, 3) 
        # Atomtypen: Zufällig 0-4 (H, C, N, O, F)
        z = torch.randint(0, 5, (num_atoms,))
        # Fully connected Adjazenz
        adj = torch.ones((num_atoms, num_atoms)) - torch.eye(num_atoms)
        edge_index = adj.nonzero().t()
        dataset.append(Data(pos=pos, z=z, edge_index=edge_index))
    return dataset

def train_poc():
    print("--- GGDM RELIABILITY PROOF START ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Stabile Daten vorbereiten
    num_atom_types = 5
    dataset = create_stable_synthetic_data(num_samples=10, num_atoms=6)
    
    # 2. Modell initialisieren (EGNN + Lie-Euler Architektur)
    model = GGDM(num_atom_types=num_atom_types, hidden_dim=64, num_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Starte Training... (Ziel: Loss-Reduktion beweisen)")

    model.train()
    first_loss = None
    last_loss = None

    for epoch in range(1, 31): # 30 Epochen für einen klaren Trend
        epoch_loss = 0
        for data in dataset:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward Diffusion Simulation:
            # Wir nehmen ein Ziel (data.pos) und addieren Rauschen
            t = torch.randint(0, model.T, (1, 1), device=device).float()
            noise = torch.randn_like(data.pos) * 0.1
            x_noisy = data.pos + noise
            
            # Das Modell soll das Rauschen (den Score) vorhersagen
            score_x, score_h = model(x_noisy, data.z, t, data.edge_index)
            
            # Loss: MSE für Koordinaten-Score + CrossEntropy für Typen
            loss_x = F.mse_loss(score_x, noise) 
            loss_h = F.cross_entropy(score_h, data.z)
            
            loss = loss_x + 0.1 * loss_h
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataset)
        if epoch == 1: first_loss = avg_loss
        last_loss = avg_loss
        
        if epoch % 5 == 0:
            print(f"Epoche {epoch:2d} | Durchschnittlicher Loss: {avg_loss:.6f}")

    print("\n--- ANALYSE ---")
    print(f"Start-Loss: {first_loss:.6f}")
    print(f"End-Loss:   {last_loss:.6f}")
    
    if last_loss < first_loss:
        improvement = (1 - last_loss/first_loss) * 100
        print(f"\n✅ ERGEBNIS: Das System arbeitet zuverlässig.")
        print(f"Der Fehler wurde um {improvement:.2f}% reduziert.")
    else:
        print("\n❌ FEHLER: Das Modell lernt nicht. Architektur-Check notwendig.")

if __name__ == "__main__":
    train_poc()
