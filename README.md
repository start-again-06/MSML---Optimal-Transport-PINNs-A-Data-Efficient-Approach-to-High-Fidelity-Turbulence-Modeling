# MSML Conference---Optimal-Transport-PINNs-A-Data-Efficient-Approach-to-High-Fidelity-Turbulence-Modeling

==================================================
OVERVIEW
==================================================

This repository presents a hybrid framework combining Optimal Transport-enhanced
Physics-Informed Neural Networks (OT-PINNs) and Sparse Identification of Nonlinear
Dynamical Systems (SINDy) to model high-Reynolds-number turbulence from noisy
Direct Numerical Simulation (DNS) data.

Designed for research and educational purposes, this project demonstrates how
interpretable, reduced-order models can be learned from partial and noisy
turbulence data using physics-informed deep learning.


==================================================
🧠 FEATURES
==================================================

- Time-dependent PINNs           : Learn velocity fields across multiple timesteps
- Optimal Transport Loss         : Alleviates training instability and spectral bias
- SINDy Integration              : Extract sparse, symbolic dynamics from the learned flow
- CUDA-Enabled                   : Training accelerated with GPU support
- Noisy Data Handling            : Trained effectively on 5% corrupted velocity data
- Energy Spectrum Comparison     : Validates learned models against DNS
- Modular Design                 : Easy to adapt for other PDE systems or flow data


==================================================
📁 REPOSITORY STRUCTURE
==================================================

Ot Pinns Turbulence.py          - Main training loop with OT-PINNs + SINDy  
ot_pinn_model_weights.py        - Script to save/load model weights  
u_pred_save_loader.py           - Visualize saved u predictions  
v_pred_save_loader.py           - Visualize saved v predictions  
ot_pinn_model.pt                - Saved model weights (binary)  
u_pred.npy / v_pred.npy         - Saved NumPy predictions  
data_vel (1).xlsx               - Input DNS data (2D slice)  
README.md                       - This file


==================================================
📦 INSTALLATION
==================================================

Install dependencies via pip:

    pip install torch numpy matplotlib pandas scipy scikit-learn

Requirements:
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for performance)


==================================================
⚙️ USAGE
==================================================

Train the OT-PINN model:

    python "Ot Pinns Turbulence.py"

This will generate:
- ot_pinn_model.pt      : Trained model weights
- u_pred.npy, v_pred.npy: Predicted velocity fields

Reload or export the model:

    python "ot_pinn_model_weights.py"

Visualize predictions:

    python "u_pred_save_loader.py"
    python "v_pred_save_loader.py"


==================================================
📈 RESULTS
==================================================

Performance Benchmarks:

Metric                     | OT-PINN (Ours)
-------------------------- | ---------------
Mean u-error               | ~2.1e-2
Mean v-error               | ~2.4e-2
Energy spectrum match      | ✅
Model stability            | ✅ Robust

Visual Highlights:
- Accurate reconstruction of velocity fields
- High-fidelity match in energy spectrum plots
- Sparse dynamics discovered using SINDy

(All visualizations are included in the training script)


==================================================
📚 PUBLICATIONS & CITATION
==================================================

If you find this repository useful for your research, please cite:

@article{MSML2025,  
  title  = {Optimal Transport PINNs with SINDy for Turbulence Modeling},  
  author = {Anjan Mahapatra & Nikhil Raj},  
  year   = {2025},  
  note   = {Manuscript in preparation}  
}


==================================================
🛠️ FUTURE DIRECTIONS
==================================================

- Extension to full 3D turbulence volumes
- Integration with attention-based neural PDE solvers
- Real-time training via online/streaming PINNs
- Differentiable coupling with Navier-Stokes solvers (e.g., JAX CFD)






