import torch
import torch.nn as nn

from torch.nn import LayerNorm

from embedding import InputEmbedding  # vocab_size, d_model
from mha import MultiHeadAttention  # d_model, num_heads, dropout
from fforward import FeedForward  # d_model, d_ff, dropout
from positional import PositionalEncoding  # d_model, max_seq_length, dropout=0.1

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()

        # Multi-head attention layers
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.layernorm3 = LayerNorm(d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the decoder block.
        x: Decoder input tensor.
        enc_output: Encoder output tensor.
        look_ahead_mask: Mask for the self-attention layer to prevent attending to future tokens.
        padding_mask: Mask to ignore padding tokens in the encoder output.
        """
        # Self-attention with look-ahead mask
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout(attn1)
        out1 = self.layernorm1(x + attn1)

        # Multi-head attention with encoder output and padding mask
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout(attn2)
        out2 = self.layernorm2(out1 + attn2)

        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding and positional encoding layers
        self.embedding = InputEmbedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Stack of decoder blocks
        self.dec_layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the decoder.
        x: Target sequence input tensor.
        enc_output: Encoder output tensor.
        look_ahead_mask: Mask to prevent the model from attending to future tokens in the target sequence.
        padding_mask: Mask to ignore padding tokens in the encoder output.
        """
        # Apply embedding and positional encoding
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # Scale embeddings
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through each decoder block
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

        return x, block1, block2  # Returns the final output and attention weights
