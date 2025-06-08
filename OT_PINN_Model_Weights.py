import torch
import torch.nn as nn

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

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINN([3, 128, 128, 128, 2]).to(DEVICE)
    model.load_state_dict(torch.load('ot_pinn_model.pt', map_location=DEVICE))
    model.eval()

    torch.save(model.state_dict(), 'ot_pinn_model.pt')
    print("Model weights successfully saved to 'ot_pinn_model.pt'")
