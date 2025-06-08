import numpy as np
import matplotlib.pyplot as plt

def load_u_prediction(file_path='u_pred.npy'):
    try:
        u_pred = np.load(file_path)
        print(f"Loaded u_pred with shape: {u_pred.shape}")
        return u_pred
    except FileNotFoundError:
        print("File not found. Ensure 'u_pred.npy' exists in the directory.")
        return None

def plot_u_field(u_pred, nx=1024, ny=1024):
    if u_pred is None:
        return
    u_grid = u_pred[:nx*ny].reshape(ny, nx)
    plt.figure(figsize=(6, 5))
    plt.imshow(u_grid, cmap='viridis', origin='lower')
    plt.colorbar(label='Predicted u')
    plt.title('Predicted u Field from OT-PINN')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    u_pred = load_u_prediction()
    plot_u_field(u_pred, nx=1024, ny=1024)
