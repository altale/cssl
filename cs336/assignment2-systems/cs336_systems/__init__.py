import importlib.metadata
from .flashattention import FlashAttentionTriton, FlashAttentionPytorch

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = ['FlashAttentionTriton', 'FlashAttentionPytorch']