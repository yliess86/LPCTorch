import matplotlib.pyplot as plt
import timeit

from tqdm import tqdm

setup = """
import numpy as np
import torchaudio
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True

from lpctorch import LPCCoefficients

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
X              = torch.cat( [ X for i in range( {0} ) ] )

# Divide in 64 2x overlapping frames
frame_duration = .016 # 16 ms
frame_overlap  = .5
K              = 32
lpc_prep       = LPCCoefficients(
    sr,
    frame_duration,
    frame_overlap,
    order = ( K - 1 )
).eval( ){1}
"""

exec = """
alphas = lpc_prep( X{0} ).detach( ).cpu( ).numpy( )
"""

cpu  = [ ]
cuda = [ ]

for i in tqdm( range( 1, 32 ) ):
    _cpu  = min( timeit.Timer( exec.format( ''         ), setup = setup.format( i, ''         ) ).repeat( 20, 1 ) )
    _cuda = min( timeit.Timer( exec.format( '.cuda( )' ), setup = setup.format( i, '.cuda( )' ) ).repeat( 20, 1 ) )
    cpu.append( _cpu )
    cuda.append( _cuda )

fig = plt.figure( )
ax  = fig.add_subplot( 111 )
ax.plot( range( 1, 32 ), cpu, 'r', label = 'cpu'  )
ax.plot( range( 1, 32 ), cuda, 'b', label = 'cuda' )
ax.legend( )
fig.canvas.draw( )
plt.show( )
