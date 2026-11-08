"""SparseLab: inspect LASSO optimization without hiding coefficient ambiguity."""
from .core import fit, path, witness
from .oracle import enumerate_optima
from .demo import demo
from .screen import screen, screen_demo
__all__ = ['fit', 'path', 'witness', 'enumerate_optima', 'demo', 'screen', 'screen_demo']
__version__ = '0.2.0'
