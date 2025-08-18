# %%

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
# import scipy.optimize as opt
# import os
# os.chdir(os.path.dirname(__file__))

# %%

# eps = sys.float_info.epsilon
eps = np.sqrt( sys.float_info.epsilon )

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

def sph_avg_a1( w, p, max_loop = 1024, eps = sys.float_info.epsilon ):
    """ N 個の M 次元ベクトルについての球面上での荷重平均を計算する。

    Spherical Weighted Average の元論文(https://mathweb.ucsd.edu/~sbuss/ResearchWeb/spheremean/paper.pdf)の
      Algorithm A1 を実装したもの。

    Arguments:
        w : 各ベクトルごとの重み係数。 サイズは [N]
        p : 球面上の荷重平均を取る球面上のベクトルを並べた行列。 サイズは [N, M]
        max_loop : 試行する反復回数の最大値。
        eps : 収束判定を行うためのしきい値。
    Returns:
        q : 球面上の荷重平均をとった値。サイズは [M]
        v : q = v @ p を満たす重み係数。サイズは [N]
        l : 収束するまでの反復回数
    """

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

    # 念の為、 w と p について正規化
    w /= np.sum(w)
    p = p / np.linalg.norm( p, axis=1 )[:,np.newaxis]

    # q の初期値として LERP で計算した加重平均値を球面上に射影したものを用いる。
    q = w @ p
    q /= np.linalg.norm( q )

    # r は p を q における接超平面(Tangent Space)に射影したベクトル l_q(p) と q との差ベクトル r = l_q(p) - 1。
    # v は v @ p = q となる重み係数。
    r, v = update_r_v( w, p, q )

    for l in range( max_loop ):
        # u は接超平面上の q の更新ベクトル。
        u = w @ r
        norm_u = np.linalg.norm( u )
        if norm_u < eps:
            break
        # q の更新には u をそのまま用いるのではなく、q+u をもとの球面に射影しなおして新たな q とする。
        q = update_q( q, u )
        # q が変わると接超平面の取り方も変わるので、 r と v の再計算を行う。
        r, v = update_r_v( w, p, q )
    return q, v, l

def sph_avg_gradient_descent( w, p, step_size = 1.0, max_loop = 1024, eps = sys.float_info.epsilon ):
    """ N 個の M 次元ベクトルについての球面上での荷重平均を計算する。
    Riemann 多様体上の最急降下法で実装したもの。

    Arguments:
        w : 各ベクトルごとの重み係数。 サイズは [N]
        p : 球面上の荷重平均を取る球面上のベクトルを並べた行列。 サイズは [N, M]
        step_size : 最急降下法のステップサイズ
        max_loop : 試行する反復回数の最大値。
        eps : 収束判定を行うためのしきい値。
    Returns:
        q : 球面上の荷重平均をとった値。サイズは [M]
        v : q = v @ p を満たす重み係数。サイズは [N]
        l : 収束するまでの反復回数
    """

    # 念の為、 w と p について正規化
    w = w / np.sum(w)
    p = p / np.linalg.norm( p, axis=1 )[:,np.newaxis]

    # q の初期値として LERP で計算した加重平均値を球面上に射影したものを用いる。
    q = w @ p
    q /= np.linalg.norm( q )

    for l in range(max_loop):
        cos_th = p @ q
        th =  np.arccos( cos_th )
        inv_sinc_th = 1.0/ np.sinc(th / np.pi)

        # v @ p = q となる重み係数 v を算出。
        v = w * inv_sinc_th/ ( np.sum(w * cos_th * inv_sinc_th) )
        
        # 勾配を算出
        grad = - ( w * 2 * th / np.sqrt( 1 - cos_th * cos_th ) ) @ p
        # 勾配について、点 q における超接平面への直交投影を計算する
        d = grad - ( q @ grad ) * q
        if np.linalg.norm( d ) < eps or np.sum( np.isnan( d )):
            break

        # 点 q における超接平面への直交投影した勾配で q を更新する
        q = q - step_size * d
        # 超球面からはみ出した q を超球面上に引っ張り戻す
        q /= np.linalg.norm( q )
    return q, v, l

def sph_avg_l_bfgs( w, p, step_size = 1.0, max_mem = 4, max_loop = 1024, eps = sys.float_info.epsilon ):
    """ N 個の M 次元ベクトルについての球面上での荷重平均を計算する。
    Riemann 多様体上の L-BFGS 法で実装したもの。

    Arguments:
        w : 各ベクトルごとの重み係数。 サイズは [N]
        p : 球面上の荷重平均を取る球面上のベクトルを並べた行列。 サイズは [N, M]
        step_size : 最急降下法のステップサイズ
        max_mem : 過去の更新履歴を記録する最大記録数
        max_loop : 試行する反復回数の最大値。
        eps : 収束判定を行うためのしきい値。
    Returns:
        q : 球面上の荷重平均をとった値。サイズは [M]
        v : q = v @ p を満たす重み係数。サイズは [N]
        l : 収束するまでの反復回数
    """
    N, M = p.shape

    # 念の為、 w と p について正規化
    w = w / np.sum(w)
    p = p / np.linalg.norm( p, axis=1 )[:,np.newaxis]

    # q の初期値として LERP で計算した加重平均値を球面上に射影したものを用いる。
    q = w @ p
    q /= np.linalg.norm( q )

    # 過去の更新履歴の記憶領域確保
    s = np.zeros( [max_mem, M ] )
    y = np.zeros( [max_mem, M ] )
    rho = np.zeros( [max_mem] )
    alpha = np.zeros([max_mem] )
    idx = 0

    cos_th = p @ q
    th =  np.arccos( cos_th )
    sinc_th = np.sinc(th / np.pi)

    # 勾配を算出
    grad = - ( w * 2 * th / np.sqrt( 1 - cos_th * cos_th ) ) @ p
    gamma = 1.0

    for l in range(max_loop):

        n_mem = min(l,max_mem)

        # v @ p = q となる重み係数 v を算出。
        v = w / ( sinc_th * np.sum(w * cos_th / sinc_th) )

        # 勾配について、点 q における超接平面への直交投影を計算する
        grad = grad - ( q @ grad ) * q
        d = np.copy(grad)

        # 加工の更新履歴をもとに、更新ベクトル d を修正
        for m in range( n_mem ):
            i = idx-m-1
            alpha[i] = rho[i] * (s[i] @ d)
            d -= alpha[i]*y[i]
        d = gamma * d
        for m in range( n_mem ):
            i = idx+m-n_mem
            beta = rho[i] * (y[i] @ d)
            d += s[i]*( alpha[i] - beta )

        if np.linalg.norm( d ) < eps or np.sum( np.isnan( d )):
            break
        
        # q の更新
        q_next = q - step_size * d
        # 超球面からはみ出した q を超球面上に引っ張り戻す
        q_next /= np.linalg.norm( q_next )

        cos_th = p @ q_next
        th =  np.arccos( cos_th )
        sinc_th = np.sinc(th / np.pi)

        grad_next = - ( w * 2 * th / np.sqrt( 1 - cos_th * cos_th ) ) @ p

        # 更新記録の計算と保存
        s[idx] = q_next - q
        y[idx] = grad_next - grad
        y[idx] -= ( q_next @ y[idx] ) * q_next

        gamma = s[idx] @ y[idx]
        rho[idx] = 1.0 / gamma
        gamma /= y[idx] @ y[idx]

        idx += 1
        if idx >= max_mem:
            idx = 0


        q = q_next
        grad = grad_next

    return q, v, l


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
        q_sph_avg, w_sph_avg, losses_sph_avg = sph_avg_a1( w, p )
        q_sdr.append( np.linalg.norm( q_sph_avg - q_slerp ) ) # / np.linalg.norm( q_slerp )) 
        w_sdr.append( np.linalg.norm( w_slerp - w_sph_avg ))
    print( "compare to slerp")
    print( "\tmax difference of q : {0:e}".format( np.max( q_sdr ) ) )
    print( "\tmax difference of w : {0:e}".format( np.max( w_sdr ) ) )

    # %%
    ## Verify the calculation time and error of the Spherical Average itself.
    N = 256
    M = 256


    w = np.random.rand( N )
    w /= np.sum( w )

    p = np.random.randn( N, M )
    p = p / np.linalg.norm( p, axis=1 )[:,np.newaxis]

    start_time = time.perf_counter()
    q, v, loss = sph_avg_a1( w, p, eps=eps )
    end_time = time.perf_counter()

    # print( (q, v, loss) )
    print( "Algorithm A1 performance check")
    print( "\tsdr of ( v @ p ) and q : {0:e}".format( np.linalg.norm( v @ p - q ) / np.linalg.norm( q ) ) )
    print( "\tloop count : {0:d}".format( l ) )
    print( "\telapsed time : {0:.2f} ms".format( (end_time - start_time)*1000 ) )

    start_time = time.perf_counter()
    q, v, loss = sph_avg_gradient_descent( w, p, eps=eps )
    end_time = time.perf_counter()

    # print( (q, v, loss) )
    print( "gradient descent performance check")
    print( "\tsdr of ( v @ p ) and q : {0:e}".format( np.linalg.norm( v @ p - q ) / np.linalg.norm( q ) ) )
    print( "\tloop count : {0:d}".format( len( loss ) ) )
    print( "\telapsed time : {0:.2f} ms".format( (end_time - start_time)*1000 ) )


    start_time = time.perf_counter()
    q, v, loss = sph_avg_l_bfgs( w, p, eps=eps )
    end_time = time.perf_counter()

    # print( (q, v, loss) )
    print( "L-BFGS performance check")
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

    L = 128
    start_time = time.perf_counter()
    for _ in range(L):
        q, v, loss = sph_avg_a1( w, p, eps=eps )
    end_time = time.perf_counter()
    np.savetxt( "v_python.csv", v, delimiter="")
    print( "num loop : {0:d}".format(len(loss)))
    print( "avg. elapsed time : {0:.2f} ms".format( (end_time - start_time)*1000/L ) )
    print( " sdr of v_cpp vs v_python : {0:e}".format( np.linalg.norm( v_cpp - v ) / np.linalg.norm( v_cpp ) ) )
    print( " sdr of q_cpp vs q_python : {0:e}".format( np.linalg.norm( v_cpp @ p - q ) ) )

    # %%

    start_time = time.perf_counter()
    for _ in range(L):
        q_gd, v_gd, loss_gd = sph_avg_gradient_descent( w, p, eps=eps )
    end_time = time.perf_counter()
    np.savetxt( "v_grad_decent.csv", v_gd, delimiter="")
    print( "num loop : {0:d}".format(len(loss_gd)))
    print( "avg. elapsed time : {0:.2f} ms".format( (end_time - start_time)*1000/L ) )
    print( " sdr of v_cpp vs v_gd : {0:e}".format( np.linalg.norm( v_cpp - v_gd ) / np.linalg.norm( v_cpp ) ) )
    print( " sdr of q_cpp vs q_gd : {0:e}".format( np.linalg.norm( v_cpp @ p - q_gd ) ) )
    # %%

    start_time = time.perf_counter()
    for _ in range(L):
        q_lbfgs, v_lbfgs, loss_lbfgs = sph_avg_l_bfgs( w, p, max_mem=8, eps=eps )
    end_time = time.perf_counter()
    np.savetxt( "v_lbfgs.csv", v_lbfgs, delimiter="")
    print( "num loop : {0:d}".format(len(loss_lbfgs)))
    print( "avg. elapsed time : {0:.2f} ms".format( (end_time - start_time)*1000/L) )
    print( " sdr of v_cpp vs v_lbfgs : {0:e}".format( np.linalg.norm( v_cpp - v_lbfgs ) / np.linalg.norm( v_cpp ) ) )
    print( " sdr of q_cpp vs q_lbfgs : {0:e}".format( np.linalg.norm( v_cpp @ p - q_lbfgs ) ) )

    # %%
