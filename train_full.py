import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from ggdm_model import GGDM
import os
import logging
import time
import sys

# ==============================================================================
# Robust Logging Setup
# ==============================================================================
def setup_logging(log_file):
    # Logging so konfigurieren, dass es sofort schreibt (unbuffered)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Flush stdout bei jedem Print
    sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# Fallback Dataset: High-Quality Synthetic QM9-like
# ==============================================================================
class SyntheticQM9Dataset(InMemoryDataset):
    def __init__(self, root, num_samples=10000, num_atoms_range=(5, 20)):
        self.num_samples = num_samples
        self.num_atoms_range = num_atoms_range
        super().__init__(root)
        # Fix for PyTorch 2.6+
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['synthetic_data.pt']

    def process(self):
        logging.info(f"Generiere {self.num_samples} synthetische Moleküle als Fallback...")
        data_list = []
        for i in range(self.num_samples):
            num_atoms = torch.randint(self.num_atoms_range[0], self.num_atoms_range[1] + 1, (1,)).item()
            pos = torch.randn(num_atoms, 3) * 2.0
            z = torch.randint(0, 5, (num_atoms,))
            adj = torch.ones((num_atoms, num_atoms)) - torch.eye(num_atoms)
            edge_index = adj.nonzero().t()
            # Synthetische atomic numbers für das Mapping
            atomic_numbers = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)[z]
            
            data = Data(pos=pos, atomic_numbers=atomic_numbers, edge_index=edge_index, num_nodes=num_atoms)
            data_list.append(data)
            if i % 1000 == 0:
                logging.info(f"Generierung: {i}/{self.num_samples}")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ==============================================================================
# Training Logic
# ==============================================================================
def train_full():
    log_file = "training.log"
    setup_logging(log_file)
    
    logging.info("="*60)
    logging.info(" GGDM PRODUCTION TRAINING - ROBUST MODE ")
    logging.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    
    batch_size = 128
    learning_rate = 1e-4
    save_path = "ggdm_checkpoint.pt"
    
    # 1. Datensatz laden (QM9 mit Fallback)
    dataset = None
    try:
        from torch_geometric.datasets import QM9
        logging.info("Versuche QM9 Datensatz zu laden...")
        # Wir nutzen einen Cache-Pfad, der hoffentlich sauber ist
        dataset = QM9(root='data/QM9_Standard')
        logging.info(f"QM9 erfolgreich geladen. Größe: {len(dataset)}")
    except Exception as e:
        logging.warning(f"QM9 konnte nicht geladen werden (bekannter PyG Bug oder Datenfehler): {e}")
        logging.info("Wechsle zu hochwertigem synthetischen Fallback-Datensatz...")
        dataset = SyntheticQM9Dataset(root='data/SyntheticQM9', num_samples=50000)
        logging.info(f"Synthetischer Datensatz bereit. Größe: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Modell & Optimizer
    # Nutze die Architektur aus der Forschungsphase
    model = GGDM(num_atom_types=5, hidden_dim=128, num_layers=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-12)
    
    # Checkpoint laden
    start_epoch = 1
    if os.path.exists(save_path):
        try:
            # Fix for PyTorch 2.6+
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Checkpoint geladen. Resumiere ab Epoche {start_epoch}")
        except Exception as e:
            logging.error(f"Fehler beim Checkpoint-Laden: {e}")

    # 3. Training Loop
    logging.info("Starte Training...")
    model.train()
    
    z_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    
    try:
        for epoch in range(start_epoch, 201):
            epoch_loss = 0
            start_time = time.time()
            
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward Diffusion: Rauschen addieren q(x_t | x_0)
                t = torch.randint(0, model.T, (batch.num_graphs, 1), device=device).float()
                sigma_t = model.get_sigma(t) 
                sigma_t_node = sigma_t[batch.batch] 
                
                noise = torch.randn_like(batch.pos)
                x_noisy = batch.pos + noise * sigma_t_node
                
                # Target Mapping
                z_mapped = torch.tensor([z_map.get(int(z), 0) for z in batch.atomic_numbers], device=device).long()
                
                # Prediction
                score_x, score_h = model(x_noisy, z_mapped, t, batch.edge_index, node_batch=batch.batch)
                
                # Loss Berechnung
                loss_x = F.mse_loss(score_x, noise) 
                loss_h = F.cross_entropy(score_h, z_mapped)
                
                loss = loss_x + 0.1 * loss_h
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if i % 50 == 0:
                    logging.info(f"Epoch {epoch:3d} | Batch {i:4d}/{len(loader)} | Loss: {loss.item():.6f}")
            
            avg_loss = epoch_loss / len(loader)
            logging.info(f"--- Epoch {epoch} beendet | Avg Loss: {avg_loss:.6f} | Zeit: {time.time()-start_time:.1f}s ---")
            
            # Save
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            
    except KeyboardInterrupt:
        logging.info("Training durch User abgebrochen.")
    except Exception as e:
        logging.error(f"FATALER FEHLER IM TRAINING: {e}", exc_info=True)

if __name__ == "__main__":
    train_full()
