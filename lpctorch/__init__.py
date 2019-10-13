"""The Init File for lpctorch module

The file imports all classes from the lpctorch module to expose them to
the user for easier API usage.
It provides a LPCCoefficients class as well as a LPCSlicer class in order
to apply an windowed lpc to a signal.
"""
from lpctorch.lpc import LPCCoefficients
from lpctorch.lpc import LPCSlicer
