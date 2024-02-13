# %%
import torch.nn as nn


# %%
# Model
class Network(nn.Module):
    def __init__(
        self,
        N_char: int,
        hidden_layer_sizes: list,
        dropout_p: float,
    ):
        super().__init__()
        self.dropout_p = dropout_p

        self.mlp = nn.ModuleList()
        self.mlp.append(
            nn.Sequential(
                nn.Linear(N_char, hidden_layer_sizes[0]),
                nn.Dropout1d(p=dropout_p),
                nn.ReLU(),
            )
        )
        for i, h in enumerate(hidden_layer_sizes[:-1]):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(h, hidden_layer_sizes[i + 1]),
                    nn.Dropout1d(p=dropout_p),
                    nn.ReLU(),
                )
            )
        self.mlp.append(nn.Linear(hidden_layer_sizes[-1], 1))

    def forward(self, char):
        for m in self.mlp:
            char = m.forward(char)
        return char


# # %%
# import torch

# B = 100
# C = 153
# char = torch.randn(B, C)

# model = Network(C, [32, 16, 8], 0.1)
# model.forward(char)


# %%
