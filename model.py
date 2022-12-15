import torch
import torch.nn as nn
import math

class Transformer(torch.nn.Module):
    def __init__(self, seq_len, num_heads, dim):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim),
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
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(dim, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
        )
        self.fc6 = nn.Linear(seq_len, 1)

        self.positional_encodings = torch.zeros(1, self.seq_len, self.dim)
        for pos in range(self.seq_len):
            for i in range(0, self.dim, 2):
                self.positional_encodings[0, pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.dim)))
                self.positional_encodings[0, pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / self.dim)))

    def forward(self, x):
        x += self.positional_encodings
        attention_output, _ = self.attention(x, x, x)
        xfc1 = self.fc1(attention_output)
        xfc1 += attention_output
        xfc2 = self.fc2(xfc1)
        xfc2 += xfc1
        xfc3 = self.fc3(x)
        xfc3 += xfc2
        xfc4 = self.fc4(xfc3)
        xfc4 += xfc3
        xfc4 = self.fc5(xfc4)
        return self.fc6(xfc4)
