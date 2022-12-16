import torch
import torch.nn as nn
import math
import numpy as np

class Transformer(torch.nn.Module):
    def __init__(self, token_dim, seq_len, num_heads, dim):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.token_dim = token_dim
        self.query = nn.Linear(token_dim, dim)
        self.key = nn.Linear(token_dim, dim)
        self.value = nn.Linear(token_dim, dim)
        self.attention1 = nn.MultiheadAttention(dim, num_heads, dropout=0.5)
        #self.attention2 = nn.MultiheadAttention(dim, num_heads)
        #self.attention3 = nn.MultiheadAttention(dim, num_heads)
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

        self.positional_encodings = torch.zeros(1, self.seq_len, self.token_dim)
        for pos in range(self.seq_len):
            for i in range(0, self.token_dim, 2):
                self.positional_encodings[0, pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.token_dim)))
                self.positional_encodings[0, pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / self.token_dim)))

    def forward(self, x):
        x += self.positional_encodings
        x = torch.transpose(x, 0, 1)
        print(x.shape)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_output1, _ = self.attention1(q, k, v)
        #attention_output2, _ = self.attention2(attention_output1, attention_output1, attention_output1)
        #attention_output3, _ = self.attention3(attention_output2, attention_output2, attention_output2)
        xfc1 = self.fc1(attention_output1) #change back to attention_output3
        xfc1 += attention_output1 #change back to attention_output3
        xfc2 = self.fc2(xfc1)
        xfc2 += xfc1
        xfc3 = self.fc3(xfc2)
        xfc3 += xfc2
        xfc4 = self.fc4(xfc3)
        xfc4 += xfc3
        xfc4 = self.fc5(xfc4)
        return self.fc6(xfc4)
