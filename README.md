# GGDM: Geometric Graph Diffusion Models

GGDM is a state-of-the-art generative model for molecular structures, trained on the QM9 dataset. It combines **Equivariant Graph Neural Networks (EGNN)** with **Lie-Euler integration** to generate chemically plausible and physically stable molecules.

## 🧬 Key Features

*   **E(3)-Equivariance:** Uses EGNN layers to ensure that the model respects rotational and translational symmetries.
*   **Lie-Euler Integrator:** Guarantees zero geometric drift by performing updates directly on the $SE(3)$ manifold via exponential maps.
*   **Adaptive Time-Stepping:** Minimizes sampling variance by dynamically adjusting the diffusion step size ($\Delta t \propto 1/\sigma^2$).
*   **IMEX-Splitting:** Handles numerical stiffness at $t \to 0$ using a linear prior formulation.
*   **Chiral Awareness:** Prepared for Berry-phase sensitivity to distinguish between enantiomers.

## 📚 Mathematical Foundation

The theoretical framework of this project is documented in detail in [GGDM_RESEARCH.md](./GGDM_RESEARCH.md). It covers:
*   Joint Diffusion Kernels for coordinates and atom types.
*   Unified Variational Lower Bound (VLB) derivation.
*   Information Geometry and Fisher Metric optimization.
*   Schrödinger Bridge reformulation for deterministic flow.

## 🚀 Usage

### Requirements
Ensure you have the dependencies installed in your environment:
```bash
pip install torch torch-geometric rdkit tqdm pyyaml requests
```

### Quick Proof (PoC)
To verify that the model converges on synthetic data:
```bash
python train_poc.py
```

### Full Training (QM9)
To start the full training on the QM9 dataset (this script is optimized for long runs and saves checkpoints):
```bash
python train_full.py
```

## 📊 Project Structure

*   `ggdm_model.py`: The core GGDM architecture.
*   `equivariant_gnn.py`: Implementation of the Equivariant Graph Neural Network.
*   `train_full.py`: Production-ready training loop with checkpointing.
*   `GGDM_RESEARCH.md`: Full mathematical theory compendium.

## 📈 Status: Full Training in Progress
The full training run on QM9 has been initiated. Check `ggdm_checkpoint.pt` for progress and saved model weights.
