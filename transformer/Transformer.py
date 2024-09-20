from DecoderLayer import DecoderLayer
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len=5000):
        super().__init__()
        self.decoder = DecoderLayer(vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len)

    def forward(self, x, mask=None):
        return self.decoder(x, mask)