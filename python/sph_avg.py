# %%

import numpy as np
import sys
import time
import matplotlib.pyplot as plt

# %%
def sph_log( w, p, q ):
    cos_th = p @ q
    th = np.arccos( cos_th )
    sinc_th = np.sinc(th / np.pi)
    p_prime = ( p -  q[np.newaxis,:] *cos_th[:,np.newaxis] )/ sinc_th[:,np.newaxis]
    w_prime = w / ( sinc_th * np.sum(w * cos_th / sinc_th) )
    return p_prime, w_prime
    
def sph_exp( q, u ):
    th = np.linalg.norm( u )
    return q * np.cos( th) + u * np.sinc( th / np.pi )

def sph_avg( w, p, max_loop = 1024, eps = sys.float_info.epsilon ):
    # initialization
    q = w @ p
    q /= np.linalg.norm( q )
    p_prime, w_prime = sph_log( w, p, q )
    losses = [np.linalg.norm( w_prime @ p - q )]
    for _ in range( max_loop ):
        u = w @ p_prime
        if np.linalg.norm( u ) < eps:
            break
        q = sph_exp( q, u )
        p_prime, w_prime = sph_log( w, p, q )
        losses.append( np.linalg.norm( w_prime @ p - q ) )
    return q, w_prime, losses

def slerp( w, p ):
    assert len( w ) == 2 or p.shape[0] == 2

    cos_omega = p[0,:] @ p[1,:]
    omega = np.arccos( cos_omega )
    sin_omega = np.sin( omega )
    w_slerp = np.asarray([
        np.sin( w[0] * omega ) / sin_omega,
        np.sin( w[1] * omega ) / sin_omega
    ])
    return w_slerp @ p, w_slerp

# %%

## if N=2, the Spherical Average is almost perfectly consistent with the SLERP.
N = 2
M = 256

q_sdr = []
w_sdr = []
for _ in range(1024):
    w = np.random.rand( N )
    w /= np.sum( w )

    p = np.random.randn( N, M )
    p = p / np.linalg.norm( p, axis=1 )[:,np.newaxis]

    q_slerp, w_slerp = slerp(w, p)
    q_sph_avg, w_sph_avg, losses_sph_avg = sph_avg( w, p )
    q_sdr.append( np.linalg.norm( q_sph_avg - q_slerp ) ) # / np.linalg.norm( q_slerp )) 
    w_sdr.append( np.linalg.norm( w_slerp - w_sph_avg ))
print( "compare to slerp")
print( "\tmax difference of q : {0:e}".format( np.max( q_sdr ) ) )
print( "\tmax difference of w : {0:e}".format( np.max( w_sdr ) ) )

# %%
## Verify the calculation time and error of the Spherical Average itself.
N = 128
M = 256
w = np.random.rand( N )
w /= np.sum( w )

p = np.random.randn( N, M )
p = p / np.linalg.norm( p, axis=1 )[:,np.newaxis]

start_time = time.perf_counter()
q, w_prime, loss = sph_avg( w, p )
end_time = time.perf_counter()

# print( (q, w_prime, loss) )
print( "spherical average performance check")
print( "\tsdr of (w_prime @ p - q) : {0:e}".format( np.linalg.norm( w_prime @ p - q ) / np.linalg.norm( q ) ) )
print( "\tloop count : {0:d}".format( len( loss ) ) )
print( "\telapsed time : {0:.2f} ms".format( (end_time - start_time)*1000 ) )

# %%

# plt.plot( loss )
# plt.show()
plt.plot( np.log10( loss ))
plt.show()
# %%