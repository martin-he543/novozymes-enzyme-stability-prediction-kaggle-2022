import pandas as pd
import numpy as np
from model import Transformer
import torch
from datetime import datetime

seq_len = 2000
dim = 512
token_dim = 4
num_tokens = 20
num_heads = 8
lr = 0.001
batch_size = 64
train_len = 22000
test_len = 6643

df_train = pd.read_csv("./data/train.csv", index_col="seq_id")
df_train_updates = pd.read_csv("./data/train_updates_20220929.csv", index_col="seq_id")

all_features_nan = df_train_updates.isnull().all("columns")

drop_indices = df_train_updates[all_features_nan].index
df_train = df_train.drop(index=drop_indices)

swap_ph_tm_indices = df_train_updates[~all_features_nan].index
df_train.loc[swap_ph_tm_indices, ["pH", "tm"]] = df_train_updates.loc[swap_ph_tm_indices, ["pH", "tm"]]

with open("amino_ranking.txt", "r") as f:
    amino_codes = f.read().split("\n")
embeddings = np.random.randn(20,token_dim)

df_train.drop(df_train[[len(x) > seq_len for x in df_train.protein_sequence]].index, inplace=True)
all_sequences = df_train["protein_sequence"].values
all_labels = df_train["tm"].values

model = Transformer(token_dim, seq_len, num_heads, dim)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

def get_train_batch():
    x_sequence = torch.zeros(batch_size, seq_len, token_dim)
    x_labels = torch.zeros(batch_size,1)
    indexes = np.random.randint(0,train_len, batch_size)
    for i in range(batch_size):
        x_raw = np.array([embeddings[amino_codes.index(x)] for x in all_sequences[indexes[i]]])
        x_padded = np.pad(x_raw, ((0, seq_len - x_raw.shape[0]%seq_len),(0,0)), "constant")
        x_sequence[i] = torch.tensor(x_padded)
        x_labels[i] = torch.tensor(all_labels[indexes[i]])
    return x_sequence, x_labels

def get_test_batch():
    x_sequence = torch.zeros(batch_size, seq_len, token_dim)
    x_labels = torch.zeros(batch_size,1)
    indexes = np.random.randint(train_len,train_len+test_len, batch_size)
    for i in range(batch_size):
        x_raw = np.array([embeddings[amino_codes.index(x)] for x in all_sequences[indexes[i]]])
        x_padded = np.pad(x_raw, ((0, seq_len - x_raw.shape[0]%seq_len),(0,0)), "constant")
        x_sequence[i] = torch.tensor(x_padded)
        x_labels[i] = torch.tensor(all_labels[indexes[i]])
    return x_sequence, x_labels

def save_states(model, optim, epoch):
    data_dict = {
        "model_states" : model.state_dict(),
        "optim_states" : optim.state_dict()
    }
    time_obj = datetime.now()
    path = f"./models/model.e{epoch:04}.{time_obj.month}.{time_obj.day}.{time_obj.hour}:{time_obj.minute}:{time_obj.second}.t"
    torch.save(data_dict, path)

def load_states(model, optim, path):
    data_dict = torch.load(path)
    model.load_state_dict(data_dict["model_states"])
    optim.load_state_dict(data_dict["optim_states"])
    return model, optim

total_epochs = 100
test_interval = 10
save_interval = 50
for epoch in range(total_epochs):
    model.train()
    data, labels = get_train_batch()
    
    output = model(data)
    loss = loss_fn(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % test_interval == 0:
        model.eval()
        data, labels = get_test_batch()
        output = model(data)
        eval_loss = loss_fn(output, labels)
        print('Epoch:', epoch, 'Train loss:', loss.item(), "Eval loss:", eval_loss.item())
        model.train()
    if epoch % save_interval == 0:
        save_states(model, optimizer, epoch)
