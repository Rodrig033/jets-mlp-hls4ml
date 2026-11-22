"""
MLP configurable. La misma clase construye el modelo base y cualquier variante
de la búsqueda: solo cambia el diccionario de configuración.
"""
import torch
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(64, 32, 32), activation="relu",
                 dropout=0.0, norm=None, init="default"):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # Capa de salida SIN softmax: nn.CrossEntropyLoss aplica log-softmax
        # internamente (más estable numéricamente).
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)
        self._init_weights(init, activation)

    def _init_weights(self, init, activation):
        if init == "default":  # Kaiming-uniforme de PyTorch
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == "he":
                    nonlin = "relu" if activation in ("relu", "leaky_relu") else "linear"
                    nn.init.kaiming_normal_(m.weight, nonlinearity=nonlin)
                elif init == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
