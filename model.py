import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

class Transformer(torch.nn.Module):
    def __init__(self, seq_len, num_heads, dim):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        #define layers
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.dense = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(seq_len, 1),
        )
        #generate positional encodings
        self.positional_encodings = torch.zeros(1, self.seq_len, self.dim)
        for pos in range(self.seq_len):
            for i in range(0, self.dim, 2):
                self.positional_encodings[0, pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.dim)))
                self.positional_encodings[0, pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / self.dim)))

    def forward(self, x):
        x += self.positional_encodings
        attention_output, _ = self.attention(x, x, x)
        out = self.dense(attention_output)
        return out

num_tokens = 20
num_heads = 8
dim = 512
seq_len = 2000

model = Transformer(seq_len, num_heads, dim)
model.train()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

df_train = pd.read_csv("./data/train.csv", index_col="seq_id")
df_train_updates = pd.read_csv("./data/train_updates_20220929.csv", index_col="seq_id")

all_features_nan = df_train_updates.isnull().all("columns")

drop_indices = df_train_updates[all_features_nan].index
df_train = df_train.drop(index=drop_indices)

swap_ph_tm_indices = df_train_updates[~all_features_nan].index
df_train.loc[swap_ph_tm_indices, ["pH", "tm"]] = df_train_updates.loc[swap_ph_tm_indices, ["pH", "tm"]]

with open("amino_ranking.txt","r") as f:
    amino_codes = f.read().split("\n")
embeddings = np.random.randn(20,dim)

df_train.drop(df_train[[len(x) > seq_len for x in df_train.protein_sequence]].index, inplace=True)
all_sequences = df_train["protein_sequence"].values
all_labels = df_train["tm"].values

batch_size = 64
train_len = 22000
test_len = 6643
def get_train_batch():
    x_sequence = torch.zeros(batch_size, seq_len, dim)
    x_labels = torch.zeros(batch_size,1)
    indexes = np.random.randint(0,train_len, batch_size)
    for i in range(batch_size):
        x_raw = np.array([embeddings[amino_codes.index(x)] for x in all_sequences[indexes[i]]])
        x_padded = np.pad(x_raw, ((0, seq_len - x_raw.shape[0]%seq_len),(0,0)), "constant")
        x_sequence[i] = torch.tensor(x_padded)
        x_labels[i] = torch.tensor(all_labels[indexes[i]])
    return x_sequence, x_labels
def get_test_batch():
    x_sequence = torch.zeros(batch_size, seq_len, dim)
    x_labels = torch.zeros(batch_size,1)
    indexes = np.random.randint(train_len,train_len+test_len, batch_size)
    for i in range(batch_size):
        x_raw = np.array([embeddings[amino_codes.index(x)] for x in all_sequences[indexes[i]]])
        x_padded = np.pad(x_raw, ((0, seq_len - x_raw.shape[0]%seq_len),(0,0)), "constant")
        x_sequence[i] = torch.tensor(x_padded)
        x_labels[i] = torch.tensor(all_labels[indexes[i]])
    return x_sequence, x_labels

total_epochs = 10
for epoch in range(total_epochs):
    data, labels = get_train_batch()
    output = model(data)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch:', epoch, 'Loss:', loss.item())
