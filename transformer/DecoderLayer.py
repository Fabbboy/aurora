import torch.nn as nn
from .InputEmbedding import InputEmbeddings
from .PositionalEncoding import PositionalEncoding
from .DecoderBlock import DecoderBlock

class DecoderLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

def test_decoder_layer():
    import torch
    decoder = DecoderLayer(10, 512, 6, 8, 2048)
    x = torch.randint(0, 10, (1, 40))
    out = decoder(x)
    assert out.shape == (1, 40, 10), f"Wrong shape: {out.shape}"
    print('DecoderLayer test passed.') # DecoderLayer test passed.

if __name__ == "__main__":
    test_decoder_layer()