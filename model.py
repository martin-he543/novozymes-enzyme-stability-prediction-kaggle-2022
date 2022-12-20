import torch
import torch.nn as nn
import numpy as np

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, fc_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc_layer = nn.Sequential(
            nn.Linear(dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        add_norm1 = self.norm1(x + attention_output)
        fcx = self.fc_layer(add_norm1)
        add_norm2 = self.norm2(add_norm1 + fcx)
        return add_norm2

class Transformer(nn.Module):
    def __init__(self, token_dim, seq_len, num_heads, dim):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.token_dim = token_dim
        self.scaling = nn.Linear(token_dim, dim)
        self.attention1 = EncoderBlock(dim, num_heads, fc_dim=dim*2, dropout=0.5)
        self.attention2 = EncoderBlock(dim, num_heads, fc_dim=dim*2, dropout=0.5)
        self.attention3 = EncoderBlock(dim, num_heads, fc_dim=dim*2, dropout=0.5)

        self.fc1 = nn.Sequential(
            nn.Linear(seq_len, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(dim, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
        )

        self.positional_encodings = torch.zeros(1, self.seq_len, self.token_dim)
        for pos in range(self.seq_len):
            for i in range(0, self.token_dim, 2):
                self.positional_encodings[0, pos, i] = \
                    np.sin(pos / (10000 ** ((2 * i) / self.token_dim)))
                self.positional_encodings[0, pos, i + 1] = \
                    np.cos(pos / (10000 ** ((2 * i) / self.token_dim)))

    def forward(self, x):
        x += self.positional_encodings
        x = self.scaling(x)
        attention_output1 = self.attention1(x)
        attention_output2 = self.attention2(attention_output1)
        attention_output3 = self.attention3(attention_output2)
        attention_output3 = torch.transpose(attention_output3, 1, 2)
        xfc1 = self.fc1(attention_output3)
        xfc1 = torch.transpose(xfc1, 1, 2)
        xfc2 = self.fc2(xfc1)
        xfc2 += xfc1
        xfc3 = self.fc3(xfc2)
        xfc3 += xfc2
        return self.fc4(xfc3)
