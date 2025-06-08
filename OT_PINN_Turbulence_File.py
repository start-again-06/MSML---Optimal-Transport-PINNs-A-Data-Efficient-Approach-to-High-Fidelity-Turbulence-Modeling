import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import grad
from sklearn.linear_model import Lasso
from scipy.fft import fft2, fftshift
import os
import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {DEVICE}")

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        return self.linears[-1](x)

class OptimalTransportLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))

def ns_residuals(x, y, t, u, v, nu):
    def grads(f, var):
        return grad(f, var, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    u_t = grads(u, t)
    u_x = grads(u, x)
    u_y = grads(u, y)
    u_xx = grads(u_x, x)
    u_yy = grads(u_y, y)

    v_t = grads(v, t)
    v_x = grads(v, x)
    v_y = grads(v, y)
    v_xx = grads(v_x, x)
    v_yy = grads(v_y, y)

    f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)

    return f_u, f_v

def build_library(data, poly_order=2):
    X = [np.ones((data.shape[0], 1))]
    for i in range(1, poly_order + 1):
        for j in range(i + 1):
            term = (data[:, 0]**(i - j) * data[:, 1]**j).reshape(-1, 1)
            X.append(term)
    return np.hstack(X)

def sindy_fit(X, dXdt, alpha=1e-3):
    model = Lasso(alpha=alpha, fit_intercept=False)
    model.fit(X, dXdt)
    return model.coef_

def load_excel_data(path):
    df = pd.read_excel(path)
    x = df['x'].values
    y = df['y'].values
    u = df['u'].values
    v = df['v'].values
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    u_grid = u.reshape((ny, nx))
    v_grid = v.reshape((ny, nx))
    return x, y, u, v, u_grid, v_grid

def compute_energy_spectrum(u, v):
    u_hat = fftshift(fft2(u))
    v_hat = fftshift(fft2(v))
    E_k = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
    return np.log10(1 + E_k)

def plot_energy_spectra(E_dns, E_pred):
    plt.figure(figsize=(12, 5))
    for i, (E, title) in enumerate(zip([E_dns, E_pred], ["DNS", "OT-PINN"])):
        plt.subplot(1, 2, i + 1)
        plt.imshow(E, cmap='inferno')
        plt.title(f"{title} Energy Spectrum")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

def diagnostics(epoch, loss):
    if epoch % 100 == 0:
        print(f"[{datetime.datetime.now()}] Epoch {epoch} | Loss: {loss:.6e}")

def train():
    data_path = '/mnt/data/data_vel (1).xlsx'
    if not os.path.exists(data_path):
        raise FileNotFoundError("Excel DNS data not found.")

    x_np, y_np, u_np, v_np, u2d, v2d = load_excel_data(data_path)

   
    timesteps = 10
    T = 1.0
    t_grid = np.linspace(0, T, timesteps)

    coords_all = []
    u_all = []
    v_all = []
    for t_val in t_grid:
        coords_all.append(np.stack([x_np, y_np, np.full_like(x_np, t_val)], axis=1))
        u_all.append(u_np * (1 + 0.05 * np.random.randn(*u_np.shape)))
        v_all.append(v_np * (1 + 0.05 * np.random.randn(*v_np.shape)))

    coords_np = np.vstack(coords_all)
    u_noisy = np.concatenate(u_all)
    v_noisy = np.concatenate(v_all)

    coords = torch.tensor(coords_np, dtype=torch.float32, requires_grad=True).to(DEVICE)
    u_true = torch.tensor(u_noisy.reshape(-1, 1), dtype=torch.float32).to(DEVICE)
    v_true = torch.tensor(v_noisy.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

   
    model = PINN([3, 128, 128, 128, 2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = OptimalTransportLoss()

    for epoch in range(3000):
        optimizer.zero_grad()
        pred = model(coords)
        u_pred, v_pred = pred[:, 0:1], pred[:, 1:2]

        f_u, f_v = ns_residuals(x, y, t, u_pred, v_pred, nu=0.01)

        loss_data = loss_fn(u_pred, u_true) + loss_fn(v_pred, v_true)
        loss_phys = torch.mean(f_u**2) + torch.mean(f_v**2)
        total_loss = loss_data + loss_phys

        total_loss.backward()
        optimizer.step()
        diagnostics(epoch, total_loss.item())

    with torch.no_grad():
        pred_final = model(coords)
        u_out = pred_final[:, 0:1].cpu().numpy()
        v_out = pred_final[:, 1:2].cpu().numpy()

        dudt = grad(pred_final[:, 0:1], t, grad_outputs=torch.ones_like(pred_final[:, 0:1]), create_graph=False)[0].cpu().numpy()

        data = np.hstack((u_out, v_out))
        X_lib = build_library(data)
        coeffs = sindy_fit(X_lib, dudt)

        print("\nSINDy Coefficients:")
        print(coeffs)

    nx = len(np.unique(x_np))
    ny = len(np.unique(y_np))
    u_recon = u_out[:nx*ny].reshape(ny, nx)  # use first time step
    v_recon = v_out[:nx*ny].reshape(ny, nx)

    E_dns = compute_energy_spectrum(u2d, v2d)
    E_pred = compute_energy_spectrum(u_recon, v_recon)
    plot_energy_spectra(E_dns, E_pred)

    torch.save(model.state_dict(), 'ot_pinn_model.pt')
    np.save('u_pred.npy', u_out)
    np.save('v_pred.npy', v_out)

if __name__ == "__main__":
    train()
