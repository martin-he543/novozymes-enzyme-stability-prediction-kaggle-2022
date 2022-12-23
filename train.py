import pandas as pd
import numpy as np
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
from model import Transformer

#set sequence constants
seq_len = 2000
token_dim = 2
num_tokens = 20

#set train-test split
train_len = 25000
test_len = 3643

#set transformer properties
dim = 128
num_heads = 4
dropout = 0.5

#set 
lr = 0.0005
batch_size = 16

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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

model = Transformer(token_dim, seq_len, num_heads, dim, device, dropout)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
loss_fn = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1000)

def get_train_batch():
    x_sequence = torch.zeros(batch_size, seq_len, token_dim).to(device)
    x_labels = torch.zeros(batch_size, 1, 1).to(device)
    indexes = np.random.randint(0,train_len, batch_size)

    for i in range(batch_size):
        x_raw = np.array([embeddings[amino_codes.index(x)] \
            for x in all_sequences[indexes[i]]])
        x_padded = np.pad(x_raw, ((0, seq_len - x_raw.shape[0]%seq_len),(0,0)),\
             "constant")
        x_sequence[i] = torch.tensor(x_padded)
        x_labels[i] = torch.tensor(all_labels[indexes[i]])
    return x_sequence, x_labels

def get_test_batch():
    x_sequence = torch.zeros(batch_size, seq_len, token_dim).to(device)
    x_labels = torch.zeros(batch_size, 1, 1).to(device)
    indexes = np.random.randint(train_len,train_len+test_len, batch_size)
    for i in range(batch_size):
        x_raw = np.array([embeddings[amino_codes.index(x)] \
            for x in all_sequences[indexes[i]]])
        x_padded = np.pad(x_raw, ((0, seq_len - x_raw.shape[0]%seq_len),(0,0)), "constant")
        x_sequence[i] = torch.tensor(x_padded)
        x_labels[i] = torch.tensor(all_labels[indexes[i]])
    return x_sequence, x_labels

def save_states(model, optim, epoch, train_loss, path):
    data_dict = {
        "model_states" : model.state_dict(),
        "optim_states" : optim.state_dict(),
        "epoch" : epoch,
        "train_loss": train_loss
    }
    torch.save(data_dict, path)

#tensorboard
log_dir = './runs/{:%Y.%m.%d.%H.%M.%S}'.format(datetime.datetime.now())
writer = SummaryWriter(log_dir=log_dir)

resume_epoch = 0
#comment the following to not resume

load_dict = torch.load("./models/2022.12.22.18.06.38.torch")
model.load_state_dict(load_dict["model_states"])
optimizer.load_state_dict(load_dict["optim_states"])
resume_epoch = load_dict["epoch"]
for g in optimizer.param_groups:
    g['lr'] = lr

total_epochs = 100000
train_batches = 100
eval_batches = 10
best_loss = 10000
best_saves = []
last_saves = []
for epoch in range(resume_epoch, total_epochs):
    model.train()
    optimizer.zero_grad()
    train_loss = 0
    for _ in range(train_batches):
        data, labels = get_train_batch()
        output = model(data)
        loss = loss_fn(output, labels)
        train_loss += loss.item()
        loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    train_loss /= train_batches
    model.eval()
    eval_loss = 0
    for _ in range(eval_batches):
        data, labels = get_test_batch()
        output = model(data)
        loss = loss_fn(output, labels)
        eval_loss += loss.item()
    eval_loss /= eval_batches
    scheduler.step(train_loss)
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    writer.add_scalar('Train loss', round(train_loss,4), global_step=epoch)
    writer.add_scalar('Eval loss', round(eval_loss,4), global_step=epoch)
    writer.add_scalar('Learning rate', current_lr, global_step=epoch)
    print('Epoch:', epoch, 'Train loss:', train_loss, "Eval loss:", eval_loss)
    timestamp = '{:%Y.%m.%d.%H.%M.%S}'.format(datetime.datetime.now())
    path = f"./models/{timestamp}.torch"
    if train_loss < best_loss:
        best_saves.append(path)
        if len(best_saves) > 5:
            os.remove(best_saves[0])
            del(best_saves[0])
        best_loss = train_loss
    else:
        last_saves.append(path)
        if len(last_saves) > 5:
            os.remove(last_saves[0])
            del(last_saves[0])
    save_states(model, optimizer, epoch, train_loss, path)
