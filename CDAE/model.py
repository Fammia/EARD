import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


class CDAE(nn.Module):
    def __init__(self, users_num: int, items_num: int, hidden_dim: int, dropout: float):
        super(CDAE, self).__init__()
        self.users_num = users_num
        self.items_num = items_num
        self.hidden_dim = hidden_dim
        self.corruption_ratio = dropout
        self.drop_out = nn.Dropout(p=dropout)

        self.user_embedding = nn.Embedding(users_num, hidden_dim)
        self.encoder = nn.Linear(items_num, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, items_num)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)

    def forward(self, user_idx, x):
        # apply corruption
        x_corrupted = self.drop_out(x)
        encoder = self.relu(self.encoder(x_corrupted) + self.user_embedding(user_idx))
        return self.decoder(encoder)
