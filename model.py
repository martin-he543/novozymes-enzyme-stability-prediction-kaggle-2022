import torch
import torch.nn as nn
import numpy as np

def positional_encodings(n_positions, d_model):
    """Generates positional encodings for a given number of positions and model depth.
    
    Args:
        n_positions: The number of positions for which to generate encodings.
        d_model: The model depth. 
        
    Returns:
        A numpy array of shape (n_positions, d_model) containing the positional encodings.
    """
    positions = np.arange(n_positions)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    
    encodings = np.zeros((n_positions, d_model))
    encodings[:, 0::2] = np.sin(positions / 10000**(dims[:, 0::2] / d_model))
    encodings[:, 1::2] = np.cos(positions / 10000**((dims[:, 1::2] + 1) / d_model))
    
    return torch.from_numpy(encodings)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x, pH):
        batch_size, seq_len, d_model = x.size()

        # Split the input into multiple heads
        queries = self.query_linear(x).view(batch_size, seq_len, self.num_heads, self.depth).permute(0, 2, 1, 3)
        keys = self.key_linear(x).view(batch_size, seq_len, self.num_heads, self.depth).permute(0, 2, 1, 3)
        values = self.value_linear(x).view(batch_size, seq_len, self.num_heads, self.depth).permute(0, 2, 1, 3)

        # Calculate dot product attention
        attention = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / self.depth**0.5
        
        #add pH to dot attention
        if pH != None:
            pH = pH.unsqueeze(-1).repeat(1, self.num_heads, seq_len, seq_len)
            attention += pH
        
        attention = attention.softmax(dim=-1)
        attention = torch.matmul(attention, values)

        # Concatenate and project the attention back to the original d_model dimension
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        attention = self.output_linear(attention)
        attention = self.dropout(attention)

        return attention

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, fc_dim, dropout):
        super().__init__()
        self.d_model = d_model
        self.attention = MultiheadAttention(d_model, num_heads, 0.35)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc_layer = nn.Sequential(
            nn.Linear(d_model, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x, pH=None):
        attention_output = self.attention(x, pH)
        add_norm1 = self.norm1(attention_output + x)
        fcx = self.fc_layer(add_norm1)
        add_norm2 = self.norm2(add_norm1 + fcx)
        return add_norm2

class Transformer(nn.Module):
    def __init__(self, seq_len, num_heads, d_model, device, dropout=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.device = device
        self.attention1 = AttentionBlock(d_model, num_heads, fc_dim=d_model*2, dropout=dropout)
        self.attention2 = AttentionBlock(d_model, num_heads, fc_dim=d_model*2, dropout=dropout)
        self.attention3 = AttentionBlock(d_model, num_heads, fc_dim=d_model*2, dropout=dropout)
        self.attention4 = AttentionBlock(d_model, num_heads, fc_dim=d_model*2, dropout=dropout)
        self.attention5 = AttentionBlock(d_model, num_heads, fc_dim=d_model*2, dropout=dropout)

        self.fc1 = nn.Sequential(
            nn.Linear(seq_len, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc4 = nn.Linear(d_model, 1)

        self.positional_encodings = positional_encodings(seq_len, d_model).to(self.device)

    def forward(self, x, pH):
        x += self.positional_encodings
        x = self.attention1(x, pH) + x
        x = self.attention2(x) + x
        x = self.attention3(x) + x
        x = self.attention4(x) + x
        x = self.attention5(x) + x
        x = torch.transpose(x, 1, 2)
        x = self.fc1(x)
        x = torch.transpose(x, 1, 2)
        x = self.fc2(x) + x
        x = self.fc3(x) + x
        return self.fc4(x)

