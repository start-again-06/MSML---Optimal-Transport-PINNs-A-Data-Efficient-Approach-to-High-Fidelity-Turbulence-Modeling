# MSML Conference---Optimal-Transport-PINNs-A-Data-Efficient-Approach-to-High-Fidelity-Turbulence-Modeling

🚀Overview
This repository presents a hybrid framework combining Optimal Transport-enhanced Physics-Informed Neural Networks (OT-PINNs) and Sparse Identification of Nonlinear Dynamical Systems (SINDy) to model high-Reynolds-number turbulence from noisy Direct Numerical Simulation (DNS) data.

Designed for research and educational purposes, this project demonstrates how interpretable, reduced-order models can be learned from partial and noisy turbulence data using physics-informed deep learning.

🧠 Features
🔁 Time-dependent PINNs: Learn velocity fields across multiple timesteps.

🎯 Optimal Transport Loss: Alleviates training instability and spectral bias.

🧩 SINDy Integration: Extract sparse, symbolic dynamics from the learned flow.

⚡ CUDA-Enabled: Training accelerated with GPU support.

🔧 Noisy Data Handling: Trained effectively on 5% corrupted velocity data.

📊 Energy Spectrum Comparison: Validates learned models against DNS.

📦 Modular Design: Easy to adapt for other PDE systems or flow data.

📁 Repository Structure

├── Ot Pinns Turbulence.py         # Main training loop with OT-PINNs + SINDy

├── ot_pinn_model_weights.py       # Script to save/load model weights

├── u_pred_save_loader.py          # Visualize saved u predictions

├── v_pred_save_loader.py          # Visualize saved v predictions

├── ot_pinn_model.pt               # Saved model weights (binary)

├── u_pred.npy / v_pred.npy        # Saved NumPy predictions

├── data_vel (1).xlsx              # Input DNS data (2D slice)

├── README.md                      # This file

📦 Installation
Install dependencies via pip:
*pip install torch numpy matplotlib pandas scipy scikit-learn*

Ensure your Python version is 3.8+ and that a CUDA-compatible GPU is available for best performance.

⚙️ Usage

1. Train the OT-PINN model
*python "Ot Pinns Turbulence.py"*

This will generate:

*ot_pinn_model.pt*: Trained model weights

*u_pred.npy*, *v_pred.npy*: Predicted velocities

2. Reload or export the model:
*python "ot_pinn_model_weights.py"*

3. Visualize predictions:
*python "u_pred_save_loader.py"*
*python "v_pred_save_loader.py"*


📈 Results
🔬 Performance Benchmarks
| Metric                | OT-PINN (Ours) |
| --------------------- | -------------- |
| Mean u-error          | \~2.1e-2       |
| Mean v-error          | \~2.4e-2       |
| Energy spectrum match | ✅              |
| Model stability       | ✅ Robust       |

🖼 Visual Results

✅ Accurate reconstruction of velocity fields.

✅ High-fidelity match in energy spectrum plots.

✅ Sparse dynamics discovered using SINDy.

All visualizations are included in the training script.

📚 Publications & Citation
If you find this repository useful for your research, please cite:
@article{MSML2025,
  title={Optimal Transport PINNs with SINDy for Turbulence Modeling},
  author={Anjan Mahapatra & Nikhil Raj},
  year={2025},
  note={Manuscript in preparation}
}

🛠️ Future Directions:

🔄 Extension to full 3D turbulence volumes.

🔍 Integration with attention-based neural PDE solvers.

🎥 Real-time training via online/streaming PINNs.

🔄 Differentiable coupling with Navier-Stokes solvers (e.g., JAX CFD).









