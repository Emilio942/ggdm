import torch
from ggdm_model import GGDM

def test_smoke_ggdm():
    print("Starte GGDM Smoke-Test...")
    
    # Konfiguration
    num_atom_types = 5 # H, C, N, O, F
    num_atoms = 10
    hidden_dim = 64
    
    # 1. Modell initialisieren
    model = GGDM(num_atom_types=num_atom_types, hidden_dim=hidden_dim, num_layers=2)
    model.eval()
    
    # 2. Dummy Daten erstellen (1 kleines Molekül)
    # x: [num_atoms, 3]
    x = torch.randn(num_atoms, 3)
    # h: [num_atoms] (Indizes der Atomtypen)
    h = torch.randint(0, num_atom_types, (num_atoms,))
    # t: [1, 1] (Zeitpunkt t)
    t = torch.tensor([[0.5]])
    # edge_index: Fully connected graph
    adj = torch.ones((num_atoms, num_atoms)) - torch.eye(num_atoms)
    edge_index = adj.nonzero().t()
    
    print(f"Input: {num_atoms} Atome, {edge_index.size(1)} Kanten.")

    # 3. Test: Forward Pass (Training-Modus)
    print("Teste Forward Pass...")
    with torch.no_grad():
        score_x, score_h = model(x, h, t, edge_index, node_batch=None)
    
    print(f"Score X Shape: {score_x.shape} (erwartet [{num_atoms}, 3])")
    print(f"Score H Shape: {score_h.shape} (erwartet [{num_atoms}, {num_atom_types}])")
    
    assert score_x.shape == (num_atoms, 3)
    assert score_h.shape == (num_atoms, num_atom_types)
    print("✅ Forward Pass erfolgreich.")

    # 4. Test: Lie-Euler Schritt (Generierung)
    print("Teste Lie-Euler Schritt...")
    t_k = 500 # t_500 von 1000
    with torch.no_grad():
        x_next, h_logits = model.lie_euler_step(x, h, t_k, edge_index)
    
    print(f"X_next Shape: {x_next.shape}")
    assert x_next.shape == x.shape
    print("✅ Lie-Euler Schritt erfolgreich.")

    print("\n--- ALLE TESTS BESTANDEN ---")

if __name__ == "__main__":
    test_smoke_ggdm()
