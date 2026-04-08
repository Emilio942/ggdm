# GGDM: Full Mathematical Theory Compendium

## 1. Geometric Dynamics & Lie Groups (SE(3)^N)
The state $X_t$ is modeled as local frames.
- **SDE:** $d\mathbf{X}_t = \mathbf{f}(\mathbf{X}_t, t) dt + \sigma(t) \mathbf{I} d\mathbf{W}_t \circ \text{exp}_{\mathbf{X}_t}^{-1}$
- **Integrator:** Lie-Euler with Exponential Map update: $\mathbf{X}_{k+1} = \exp_{\mathbf{X}_k}(\mathbf{v}_k)$.
- **Numerical Stability:** IMEX-Splitting with Linear Prior cancels the $1/t$ singularity.
- **Error Bound:** $W_2^2(\mu_{\text{num}}, \mu_{\text{true}}) = \mathcal{O}(\Delta t)$.

## 2. Schrödinger Bridges & Optimal Transport
Reformulate GGDM as a Schrödinger Bridge Problem (SBP) for efficient, deterministic generation.
- **Dual Potentials:** $\psi_t, \hat{\psi}_t$ solve the coupled Schrödinger system.
- **Deterministic Flow (ODE):** $\frac{dY_t}{dt} = -\varepsilon \nabla [\log \psi_t + \log \hat{\psi}_t]$.
- **Symmetry:** Potentials are invariant under $E(3) \times S_N$ via relative distance kernels.

## 3. Information Geometry & Training
- **Unified VLB:** $\mathcal{L}_{\text{VLB}} = \lambda(t) \mathcal{L}_{\text{coord}} + (1 - \lambda(t)) \mathcal{L}_{\text{cat}}$
- **Loss Weighting:** $\lambda(t) = \frac{\operatorname{tr} \mathcal{I}_F(t)}{\operatorname{tr} \mathcal{I}_F(t) + \frac{1}{\eta} D_{\text{rel}}(t)}$ based on Fisher Information.
- **Mode Collapse:** Ricci curvature of the score manifold determines stability; negative curvature in hyperbolic spaces promotes exploration.

## 4. Multi-Scale Renormalization Group (RG) Flow
- **Isomorphism:** Forward diffusion is Wilsonian RG (integrating out high-frequency modes).
- **Hierarchical SDE:** Coarse variables $\mathbf{z}_t$ drive topology; fine variables $\mathbf{x}_t$ are sampled conditionally on fibers $\Pi^{-1}(\mathbf{z}_t)$.
- **Information Loss:** Quantified by the Jacobian of the geometric submersion $\Pi$.

## 5. Topological & Physical Constraints
### 5.1 Topological Score
Uses Hodge-Laplacian $\Delta_k$ and Persistent Homology.
- **Formula:** $\nabla_{x_t} \log p_{\text{topo}} = -\sum_{k} \operatorname{tr}[(\partial_{x_t} \Delta_k) \phi_k]^\top \tilde{\partial}_k^* \phi_k$

### 5.2 Chirality & Berry Phase
Distinguish enantiomers via the topological phase of the local chiral frame.
- **Berry Phase:** $\Phi = 2\pi \cdot \text{sign}(\tau)$, where $\tau = \mathbf{r}_{ij} \cdot (\mathbf{r}_{ik} \times \mathbf{r}_{il})$.
- **Chiral Correction:** $\mathcal{L}_{\text{chiral}} = \lambda_{\text{chiral}} (\sigma_{\text{model}} - \text{sign}(\tau_{\text{current}}))^2$.

## 6. Spontaneous Symmetry Breaking (SSB)
- **Phase Transition:** Critical time $t_c$ where the $SE(3)$-invariant noise bifurcates into rigid vacua.
- **Goldstone Modes:** Manifest as long-range correlations in the score field $\mathbf{s}_\theta$.
- **Kibble-Zurek Mechanism:** Predicts topological defect density based on the quench rate $\beta(t)$.

## 7. Global Stability & Operator Theory
- **Koopman Operator:** Linear evolution of observables in infinite-dimensional space.
- **Spectral Gap:** Controls the mixing time $\tau_{\text{mix}} \approx \frac{1}{\gamma} \log(\frac{1}{\varepsilon})$.
- **Lyapunov Exponent:** Bounded by the norm of the score Jacobian: $\lambda_{\max} \le \|\mathcal{D}s_\theta\| + \frac{d}{2}\beta^{-1}$.
