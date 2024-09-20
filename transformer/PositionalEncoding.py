import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        # Create the positional encoding matrix
        encoding = torch.zeros(max_len, embed_dim)  # Create encoding without gradients
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('PE', encoding.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to input
        return x + self.PE[:, :x.size(1), :].to(x.device)

def test_positional_encoding():
    positional = PositionalEncoding(512, 40)
    x = torch.randn(1, 40, 512)
    out = positional(x)
    assert out.shape == (1, 40, 512), f"Wrong shape: {out.shape}"
    print('PositionalEncoding test passed.') # PositionalEncoding test passed.

if __name__ == "__main__":
    test_positional_encoding()