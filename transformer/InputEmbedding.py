import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x) * (x.size(-1) ** 0.5)  # Scale embeddings
    
def test_input_embeddings():
    import torch
    embed = InputEmbeddings(10, 20)
    x = torch.randint(0, 10, (4, 5))
    out = embed(x)
    assert out.shape == (4, 5, 20), f"Wrong shape: {out.shape}"
    print('InputEmbeddings test passed.') # InputEmbeddings test passed.

if __name__ == "__main__":
    test_input_embeddings()