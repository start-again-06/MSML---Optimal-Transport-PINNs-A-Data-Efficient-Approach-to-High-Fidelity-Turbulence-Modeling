import numpy as np
import matplotlib.pyplot as plt

def load_v_prediction(file_path='v_pred.npy'):
    try:
        v_pred = np.load(file_path)
        print(f"Loaded v_pred with shape: {v_pred.shape}")
        return v_pred
    except FileNotFoundError:
        print("File not found. Ensure 'v_pred.npy' exists in the directory.")
        return None

def plot_v_field(v_pred, nx=1024, ny=1024):
    if v_pred is None:
        return
    v_grid = v_pred[:nx*ny].reshape(ny, nx)
    plt.figure(figsize=(6, 5))
    plt.imshow(v_grid, cmap='plasma', origin='lower')
    plt.colorbar(label='Predicted v')
    plt.title('Predicted v Field from OT-PINN')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    v_pred = load_v_prediction()
    plot_v_field(v_pred, nx=1024, ny=1024)
