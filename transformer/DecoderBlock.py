import torch.nn as nn
from .MultiHeadAttention import MultiHeadAttention
from .FeedForward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

def test_decoder_block():
    import torch
    decoder_block = DecoderBlock(512, 8, 2048)
    print(decoder_block)
    x = torch.randn(1, 40, 512)
    out = decoder_block(x)
    assert out.shape == (1, 40, 512), f"Wrong shape: {out.shape}"
    print('DecoderBlock test passed.') # DecoderBlock test passed.

if __name__ == "__main__":
    test_decoder_block()