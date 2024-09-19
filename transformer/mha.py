import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear layers for Query, Key, and Value
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Output linear layer
        self.dense = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_length, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def look_ahead_mask(size):
        """
        Create a mask to prevent attention to future tokens (look-ahead).
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    @staticmethod
    def attention(q, k, v, mask=None, dropout=None):
        """
        Scaled dot-product attention mechanism.
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.nn.functional.softmax(scores, dim=-1)

        if dropout is not None:
            attention = dropout(attention)

        return torch.matmul(attention, v), attention

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of the multi-head attention layer.
        """
        batch_size = q.size(0)

        # Apply linear layers to the input queries, keys, and values
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        # Split the output into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Calculate attention
        attention, weights = self.attention(query, key, value, mask, self.dropout)

        # Concatenate attention output
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(batch_size, -1, self.d_model)

        # Apply the final linear layer
        output = self.dense(attention)

        return output, weights
