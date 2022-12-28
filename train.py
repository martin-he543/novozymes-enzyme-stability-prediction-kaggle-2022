import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
from model import Transformer, positional_encodings
from data_loader import DataLoader
import time
from scipy.stats import spearmanr

torch.set_num_threads(12)

#set sequence constants
seq_len = 1024
num_tokens = 20

#set transformer properties
d_model = 512
num_heads = 8
dropout = 0.25

#set
lr = 0.0001
weight_decay = 0.1
batch_size = 16

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

model = Transformer(seq_len, num_heads, d_model, device, dropout).to(device)
pe = positional_encodings(seq_len, d_model).to(device)

loader = DataLoader(seq_len, d_model, batch_size, "embeddings_512.npy", "train-test.pci")
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1500)

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
load_from = ""#models/2022.12.28.07.49.39.torch"#2022.12.26.17.33.34.torch"#"./models/2022.12.26.18.37.20.torch"
if load_from != "":
    load_dict = torch.load(load_from)
    model.load_state_dict(load_dict["model_states"])
    optimizer.load_state_dict(load_dict["optim_states"])
    resume_epoch = load_dict["epoch"]
    for g in optimizer.param_groups:
        g['lr'] = lr

# 2 GPUs
# model = nn.DataParallel(model, device_ids = [1,2]).to(device)

total_epochs = 100000
train_batches = 150
eval_batches = 10
best_loss = 10000
best_saves = []
last_saves = []


for epoch in range(resume_epoch, total_epochs):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    train_loss = 0
    for _ in range(train_batches):
        seq, ph, tm = loader.get_train_batch()
        seq = seq.to(device)
        seq += pe
        ph = ph.to(device)
        tm = tm.to(device)
        output = model(seq, ph)
        loss = loss_fn(output, tm)
        train_loss += loss.item()
        loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    train_loss /= train_batches
    model.eval()
    eval_loss = 0
    x = np.ndarray(eval_batches*batch_size)
    y = np.ndarray(eval_batches*batch_size)
    for i in range(eval_batches):
        seq, ph, tm = loader.get_test_batch()
        x[i*batch_size:(i+1)*batch_size] = tm.squeeze().numpy()
        seq = seq.to(device)
        seq += pe
        ph = ph.to(device)
        tm = tm.to(device)
        output = model(seq, ph)
        loss = loss_fn(output, tm)
        eval_loss += loss.item()
        y[i*batch_size:(i+1)*batch_size] = output.detach().squeeze().cpu().numpy()
    r, _ = spearmanr(x, y)
    eval_loss /= eval_batches
    
    scheduler.step(train_loss)
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    writer.add_scalar('Train loss', train_loss, global_step=epoch)
    writer.add_scalar('Eval loss', eval_loss, global_step=epoch)
    writer.add_scalar('Learning rate', current_lr, global_step=epoch)
    writer.add_scalar('Spearman', r, global_step=epoch)
    tf = time.time() - t0
    print(f"Epoch: {epoch}\tTrain loss: {train_loss:.6f}\tEval loss: {eval_loss:.6f}t Spearman: {r:.3f}\tElapsed: {tf:.2f} s")
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
