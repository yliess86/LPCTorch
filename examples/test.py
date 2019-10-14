import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True

from lpctorch import LPCCoefficients
from librosa.core import lpc

# Load audio file
sr             = 16000 # 16 kHz
path           = './examples/sample.wav'
data, _sr      = torchaudio.load( path, normalization = lambda x: x.abs( ).max( ) )
data           = torchaudio.transforms.Resample( _sr, sr )( data )
duration       = data.size( 1 ) / sr

# Get audio sample worth of 512 ms
worth_duration = .512 # 512 ms ( 256 ms before and 256 ms after )
worth_size     = int( np.floor( worth_duration * sr ) )
X              = data[ :, :worth_size ]
X_duration     = X.size( 1 ) / sr
X              = torch.cat( [ X for i in range( 4 ) ] )

# ====================== ME ====================================================
# Divide in 64 2x overlapping frames
frame_duration = .016 # 16 ms
frame_overlap  = .5
K              = 32
lpc_prep       = LPCCoefficients(
    sr,
    frame_duration,
    frame_overlap,
    order = ( K - 1 )
).eval( ).cuda( )
alphas         = lpc_prep( X.cuda( ) ).detach( ).cpu( ).numpy( )

# Print details
print( f'[Init]   [Audio]  src: { path }, sr: { sr }, duration: { duration }' )
print( f'[Init]   [Sample] size: { X.shape }, duration: { X_duration }' )
print( f'[Me]     [Alphas] size: { alphas.shape }' )

# ====================== NOT ME ================================================
frames  = lpc_prep.frames( X.cuda( ) )
frames  = frames[ 0 ].detach( ).cpu( ).numpy( )
_alphas = np.array( [ lpc( frames[ i ], K - 1 ) for i in range( frames.shape[ 0 ] ) ] )
print( f'[Not Me] [Alphas] size: { _alphas.shape }' )

# Draw frames
fig = plt.figure( )
ax  = fig.add_subplot( 211 )
ax.imshow( alphas[ 0 ] )
ax  = fig.add_subplot( 212 )
ax.imshow( _alphas )
fig.canvas.draw( )
plt.show( )
