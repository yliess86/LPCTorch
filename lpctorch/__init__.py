"""The Init File for lpctorch module

The file imports all classes from the lpctorch module to expose them to
the user for easier API usage.
It provides a LPCCoefficients class as well as a LPCSlicer class in order
to apply an windowed lpc to a signal.
"""
from lpctorch.lpc import LPCCoefficients
from lpctorch.lpc import LPCSlicer

__name__ = "LPCTorch"
__version__ = "0.1.1"
__author__ = "yliess"
__url__ = "https://github.com/TowardHumanizedInteraction/LPCTorch"
__email__ = "hatiyliess86@gmail.com"