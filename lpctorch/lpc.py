"""LPC module from lpctorch

The file provides an access to pytorch modules for windowed
Linear Predictive Coding (LPC). The LPCSlicer turns a windowed signal
into overlapping frames and the LPCCoefficients uses it to compute the
order LPC coefficients for each frame using the Burg’s method [1]. The
implementation is a port of the librosa library algorithm [2] to support
batch and frames using pytorch operations.

[1] Larry Marple A New Autoregressive Spectrum Analysis Algorithm IEEE
    Transactions on Accoustics, Speech, and Signal Processing
    vol 28, no. 4, 1980
[2] https://librosa.github.io/librosa/_modules/librosa/core/audio.html#lpc
"""
import torch.nn as nn
import numpy as np
import torch

from torch.autograd import Variable
from typing import Any

class LPCSlicer( nn.Module ):
    """LPC Slicer

    The LPCSlicer slice a given signal into n overlapping frames.
    The DC component of the output frames is removed and output is windowed.

    Attributes
    ----------
    sr      : int
              default 16000
              Sample rate of the audio signal.
    duration: float
              default .016
              Frame duration in seconds.
    overlap : float
              default .5
              Factor of overlapping for the frames.
    window  : Any
              default torch.hann_window
              Window function to be applied to each of the frame.
    padded  : bool
              defalt False
              Do the input need to be padded to allow full windowing
    """
    def __init__(
        self    : 'LPCSlicer',
        sr      : int         = 16000,
        duration: float       = .016,
        overlap : float       = .5,
        window  : Any         = torch.hann_window,
        padded  : bool        = False
    ) -> None:
        """Init

        Parameters
        ----------
        sr      : int
                  default 16000
                  Sample rate of the audio signal.
        duration: float
                  default .016
                  Frame duration in seconds.
        overlap : float
                  default .5
                  Factor of overlapping for the frames.
        window  : Any
                  default torch.hann_window
                  Window function to be applied to each of the frame.
        padded  : bool
                  defalt False
                  Do the input need to be padded to allow full windowing
        """
        super( LPCSlicer, self ).__init__( )
        self.size    = int( np.floor( duration * sr ) )
        self.offset  = int( np.floor( self.size * overlap ) )
        self.padding = self.offset if padded else 0

        window       = Variable( window( self.size ), requires_grad = False )
        self.register_buffer( 'window', window )

    def forward( self: 'LPCSlicer', X: torch.Tensor ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Input signal to be sliced into frames.
           Expected input is [ Batch, Samples ]

        Returns
        -------
        X: torch.Tensor
           Input signal sliced into frames.
           Expected output is [ Batch, Frames, Samples ]
        """
        X  = nn.functional.pad( X, ( 0, self.padding ), 'constant', 0. )
        X  = X.unfold( -1, self.size, self.offset )
        X -= X.mean( axis = -1, keepdim = True ) # axis = 'S' )
        X *= self.window
        return X

class LPCCoefficients( nn.Module ):
    """LPC Coefficients

    The LPCCoefficients uses the output of the LPCSlicer to compute the
    Linear Predictive Coding Coefficients following the Burg's Method [1].
    The implementation is a port of the one available in the librosa audio
    library to support batch, frames and uses only pytorch operations.

    Attributes
    ----------
    sr      : int
              default 16000
              Sample rate of the audio signal.
    duration: float
              default .016
              Frame duration in seconds.
    overlap : float
              default .5
              Factor of overlapping for the frames.
    order   : int
              Number of Linear Predictive Coefficients - 1
    window  : Any
              default torch.hann_window
              Window function to be applied to each of the frame.
    padded  : bool
              defalt False
              Do the input need to be padded to allow full windowing
    """
    def __init__(
        self: 'LPCCoefficients',
        sr      : int         = 16000,
        duration: float       = .016,
        overlap : float       = .5,
        order   : int         = 31,
        window  : Any         = torch.hann_window,
        padded  : bool        = False
    ) -> None:
        """Init

        Parameters
        ----------
        sr      : int
                  default 16000
                  Sample rate of the audio signal.
        duration: floatAttributes
                  default .016
                  Frame duration in seconds.
        overlap : float
                  default .5
                  Factor of overlapping for the frames.
        order   : int
                  Number of Linear Predictive Coefficients - 1
        window  : Any
                  default torch.hann_window
                  Window function to be applied to each of the frame.
        padded  : bool
                  defalt False
                  Do the input need to be padded to allow full windowing
        """
        if order <= 1:
            raise ValueError('LPC order must be greater > 1 or it is useless')

        super( LPCCoefficients, self ).__init__( )
        self.frames = LPCSlicer( sr, duration, overlap, window, padded )
        self.order  = order
        self.p      = order + 1

    def forward( self: 'LPCCoefficients', X: torch.Tensor ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Input signal to be sliced into frames.
           Expected input is [ Batch, Samples ]

        Returns
        -------
        X: torch.Tensor
           LPC Coefficients computed from input signal after slicing.
           Expected output is [ Batch, Frames, Order + 1 ]
        """
        X                      = self.frames( X )
        B, F, S                = X.size( )

        alphas                 = torch.zeros( ( B, F, self.p ),
            dtype         = X.dtype,
            device        = X.device,
            requires_grad = False
        )
        alphas[ :, :, 0 ]      = 1.
        alphas_prev            = torch.zeros( ( B, F, self.p ),
            dtype         = X.dtype,
            device        = X.device,
            requires_grad = False
        )
        alphas_prev[ :, :, 0 ] = 1.

        fwd_error              = X[ :, :, 1:   ]
        bwd_error              = X[ :, :,  :-1 ]

        den                    = (
            ( fwd_error * fwd_error ).sum( axis = -1 ) + \
            ( bwd_error * bwd_error ).sum( axis = -1 )
        ).unsqueeze( -1 )

        for i in range( self.order ):
            dot_bfwd            = ( bwd_error * fwd_error ).sum( axis = -1 )\
                                                           .unsqueeze( -1 )

            reflect_coeff       = -2. * dot_bfwd / den
            alphas_prev, alphas = alphas, alphas_prev

            for j in range( 1, i + 2 ):
                alphas[ :, :, j ] = alphas_prev[   :, :,         j ] + \
                                    reflect_coeff[ :, :,         0 ] * \
                                    alphas_prev[   :, :, i - j + 1 ]

            fwd_error_tmp       = fwd_error
            fwd_error           = fwd_error + reflect_coeff * bwd_error
            bwd_error           = bwd_error + reflect_coeff * fwd_error_tmp

            q                   = 1. - reflect_coeff ** 2
            den                 = q * den - \
                                  bwd_error[ :, :, -1 ].unsqueeze( -1 ) ** 2 - \
                                  fwd_error[ :, :,  0 ].unsqueeze( -1 ) ** 2

            fwd_error           = fwd_error[ :, :, 1:   ]
            bwd_error           = bwd_error[ :, :,  :-1 ]

        return alphas
