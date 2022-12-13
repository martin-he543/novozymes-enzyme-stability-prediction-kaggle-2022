import torch
import torch.nn as nn
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
            nn.Linear(dim, 1)
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

# Define the dimensions of the input data
num_tokens = 20
num_heads = 8
dim = 512
seq_len = 2000

model = Transformer(seq_len, num_heads, dim)
model.train()

rand_inputs = torch.randn(1, seq_len, dim)
labels = torch.randn(1)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

total_epochs = 100
for epoch in range(total_epochs):
    output = model(rand_inputs)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('Epoch:', epoch, 'Loss:', loss.item())
