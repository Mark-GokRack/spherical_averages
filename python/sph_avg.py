# %%

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
# import scipy.optimize as opt
import os
os.chdir(os.path.dirname(__file__))

# %%
def update_r_v( w, p, q ):
    cos_th = p @ q
    th = np.arccos( cos_th )
    sinc_th_inv = 1.0 / np.sinc(th / np.pi)
    r = ( p -  q[np.newaxis,:] *cos_th[:,np.newaxis] ) * sinc_th_inv[:,np.newaxis]
    v = w * sinc_th_inv / ( np.sum( w * cos_th * sinc_th_inv ) )
    return r, v
    
def update_q( q, u ):
    th = np.linalg.norm( u )
    return q * np.cos( th) + u * np.sinc( th / np.pi )

def sph_avg( w, p, max_loop = 1024, eps = sys.float_info.epsilon ):
    # initialization
    w /= np.sum(w)
    q = w @ p
    q /= np.linalg.norm( q )
    r, v = update_r_v( w, p, q )
    losses = [np.linalg.norm( v @ p - q )]
    for _ in range( max_loop ):
        u = w @ r
        if np.linalg.norm( u ) < eps:
        #if  u @ u < eps:
            break
        q = update_q( q, u )
        r, v = update_r_v( w, p, q )
        losses.append( np.linalg.norm( v @ p - q ) )
        # losses.append( losses[-1])
    return q, v, losses

"""
def sph_avg_2( w, p, max_loop = 1024, eps = sys.float_info.epsilon ):
    # initialization
    w /= np.sum(w)
    q = w @ p
    q /= np.linalg.norm( q )
    r, v = update_r_v( w, p, q )
    losses = [np.linalg.norm( v @ p - q )]
    for _ in range( max_loop ):
        u = w @ r
        if np.linalg.norm( u ) < eps:
            break
        
        ### TBD

        q = update_q( q, u )
        r, v = update_r_v( w, p, q )
        losses.append( np.linalg.norm( v @ p - q ) )
    return q, v, losses
"""

def sph_avg_gradient_descent( w, p, max_loop = 1024, eps = sys.float_info.epsilon ):
    w /= np.sum(w)
    q = w @ p
    q /= np.linalg.norm( q )
    alpha = 1.0
    cos_th = p @ q
    th =  np.arccos( cos_th )
    if False:
        losses = [ w @ ( th * th ) ]
    else:
        _, v = update_r_v( w, p, q )
        losses = [np.linalg.norm( v @ p - q )]

    for _ in range(max_loop):
        # grad_ = - np.sum( [ w[i] * 2 * np.arccos( th[i] ) / np.sqrt(1-th[i]*th[i]) * p[i]  for i in range( p.shape[0] ) ] , axis=0 )
        d = - ( w * 2 * th / np.sqrt( 1 - cos_th * cos_th ) ) @ p
        grad = d - ( q @ d ) * q
        if np.linalg.norm( grad ) < eps or np.sum( np.isnan( grad )):
            break

        q_new = q - alpha * grad
        q_new /= np.linalg.norm( q_new )
        cos_th = p @ q_new
        th =  np.arccos( cos_th )
        if False:
            loss = w @ ( th * th )
        else:
            sinc_th = np.sinc(th / np.pi)
            v = w / ( sinc_th * np.sum(w * cos_th / sinc_th) )
            loss = np.linalg.norm( v @ p - q_new )
        if losses[-1] - loss < eps:
            break
        losses.append( loss )
        q = q_new
    _, v = update_r_v( w, p, q )
    return q, v, losses

def slerp( w, p ):
    assert len( w ) == 2 or p.shape[0] == 2

    cos_omega = p[0,:] @ p[1,:]
    omega = np.arccos( cos_omega )
    sin_omega = np.sin( omega )
    v = np.asarray([
        np.sin( w[0] * omega ) / sin_omega,
        np.sin( w[1] * omega ) / sin_omega
    ])
    return v @ p, v

# %%
if __name__ == "__main__":
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
    q, v, loss = sph_avg( w, p )
    end_time = time.perf_counter()

    # print( (q, v, loss) )
    print( "Algorithm A1 performance check")
    print( "\tsdr of ( v @ p ) and q : {0:e}".format( np.linalg.norm( v @ p - q ) / np.linalg.norm( q ) ) )
    print( "\tloop count : {0:d}".format( len( loss ) ) )
    print( "\telapsed time : {0:.2f} ms".format( (end_time - start_time)*1000 ) )

    start_time = time.perf_counter()
    q, v, loss = sph_avg_gradient_descent( w, p )
    end_time = time.perf_counter()

    # print( (q, v, loss) )
    print( "gradient descent performance check")
    print( "\tsdr of ( v @ p ) and q : {0:e}".format( np.linalg.norm( v @ p - q ) / np.linalg.norm( q ) ) )
    print( "\tloop count : {0:d}".format( len( loss ) ) )
    print( "\telapsed time : {0:.2f} ms".format( (end_time - start_time)*1000 ) )

    # %%

    # plt.plot( loss )
    # plt.plot( np.log10( loss ))
    # plt.show()
    # %%

    w = np.loadtxt("w.csv", delimiter=",")
    p = np.loadtxt("p.csv", delimiter=",")
    v_cpp = np.loadtxt("v.csv", delimiter=",")
    w /= np.sum( w )
    p /= np.linalg.norm( p, axis=1 )[:,np.newaxis]
    start_time = time.perf_counter()
    q, v, loss = sph_avg( w, p )
    end_time = time.perf_counter()
    np.savetxt( "v_python.csv", v, delimiter="")
    print( "num loop : {0:d}".format(len(loss)))
    print( "elapsed time : {0:.2f} ms".format( (end_time - start_time)*1000 ) )
    print( " sdr of v_cpp vs v_python : {0:e}".format( np.linalg.norm( v_cpp - v ) / np.linalg.norm( v_cpp ) ) )
    print( " sdr of q_cpp vs q_python : {0:e}".format( np.linalg.norm( v_cpp @ p - q ) ) )

    # %%

    start_time = time.perf_counter()
    q_gd, v_gd, loss_gd = sph_avg_gradient_descent( w, p )
    end_time = time.perf_counter()
    np.savetxt( "v_grad_decent.csv", v_gd, delimiter="")
    print( "num loop : {0:d}".format(len(loss_gd)))
    print( "elapsed time : {0:.2f} ms".format( (end_time - start_time)*1000 ) )
    print( " sdr of v_cpp vs v_gd : {0:e}".format( np.linalg.norm( v_cpp - v_gd ) / np.linalg.norm( v_cpp ) ) )
    print( " sdr of q_cpp vs q_gd : {0:e}".format( np.linalg.norm( v_cpp @ p - q_gd ) ) )

    # %%
