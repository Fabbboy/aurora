import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

def test_feed_forward():
    import torch
    ff = FeedForward(512, 2048)
    x = torch.randn(1, 40, 512)
    out = ff(x)
    assert out.shape == (1, 40, 512), f"Wrong shape: {out.shape}"
    print('FeedForward test passed.') # FeedForward test passed.

if __name__ == "__main__":
    test_feed_forward()