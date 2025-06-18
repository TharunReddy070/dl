import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        # Linear layers for Q, K, V
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, mask=None):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Linear projections and reshaping for multi-head attention
        values = self.values(value).view(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)
        return self.fc_out(out)

# Test run
embed_size = 128
heads = 8
attention = MultiHeadAttention(embed_size, heads)

x = torch.rand(2, 10, embed_size)  # (batch_size=2, seq_len=10, embed_size=128)
output = attention(x, x, x)
print("Output shape:", output.shape)  # Expected: [2, 10, 128]
