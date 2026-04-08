import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from ggdm_model import GGDM
import os

def train_full():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Nutze Device: {device}")
    
    batch_size = 64
    learning_rate = 5e-4
    save_path = "ggdm_checkpoint.pt"
    
    # 2. Voller Datensatz
    dataset = QM9(root='data/QM9')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Modell & Optimizer
    # Erhöhte Komplexität für echtes Training
    model = GGDM(num_atom_types=5, hidden_dim=128, num_layers=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-12)
    
    # Checkpoint laden, falls vorhanden
    start_epoch = 1
    if os.path.exists(save_path):
        print("Lade existierenden Checkpoint...")
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    print("Starte Langzeit-Training (QM9 Full)...")
    model.train()
    
    try:
        for epoch in range(start_epoch, 101): # Bis zu 100 Epochen
            epoch_loss = 0
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward Diffusion: Rauschen addieren
                # (Hier würde die mathematische Rausch-Logik aus GGDM_RESEARCH implementiert)
                t = torch.randint(0, model.T, (batch.num_graphs, 1), device=device).float()
                
                # Score Prediction
                # Wir mappen batch.z auf 0-4 (für H, C, N, O, F)
                z_mapped = (batch.z % 5).long()
                score_x, score_h = model(batch.pos, z_mapped, t, batch.edge_index)
                
                # Loss Berechnung (vereinfacht für diesen Entwurf)
                loss_x = F.mse_loss(score_x, torch.zeros_like(score_x))
                loss_h = F.cross_entropy(score_h, z_mapped)
                
                loss = loss_x + 0.1 * loss_h # Gewichtung nach Carlins VLB-Logik
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")
            
            # Epoch Ende: Checkpoint speichern
            print(f"--- Epoch {epoch} beendet. Durchschnittlicher Loss: {epoch_loss/len(loader):.4f} ---")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, save_path)
            
    except KeyboardInterrupt:
        print("\nTraining durch User unterbrochen. Fortschritt wurde gespeichert.")

if __name__ == "__main__":
    train_full()
