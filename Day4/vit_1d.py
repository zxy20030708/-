import torch
from torch import nn, Tensor
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from typing import Optional, Tuple, List

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation and dropout."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class Attention(nn.Module):
    """Multi-head self-attention mechanism with layer normalization."""
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if self.project_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer encoder with alternating attention and feed-forward layers."""
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x  # Residual connection
            x = ff(x) + x    # Residual connection
        return x

class ViT(nn.Module):
    """Vision Transformer for 1D time series data."""
    def __init__(self, *, seq_len: int, patch_size: int, num_classes: int, dim: int, 
                 depth: int, heads: int, mlp_dim: int, channels: int = 3, 
                 dim_head: int = 64, dropout: float = 0., emb_dropout: float = 0.) -> None:
        super().__init__()
        assert seq_len % patch_size == 0, "Sequence length must be divisible by patch size"

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series: Tensor) -> Tensor:
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x, ps = pack([cls_tokens, x], 'b * d')
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)

def test_vit() -> None:
    """Test function to verify ViT implementation."""
    v = ViT(
        seq_len=256,
        patch_size=16,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    time_series = torch.randn(4, 3, 256)
    print("Input tensor shape:", time_series.shape)
    logits = v(time_series)  # (4, 1000)
    print("Output tensor shape:", logits.shape)

if __name__ == '__main__':
    test_vit()