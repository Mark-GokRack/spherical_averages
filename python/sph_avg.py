# %%

import numpy as np
import sys
import time

# %%

def slerp( w, p ):
    """Slerp により 2 個の M 次元ベクトルについての球面上での線形補間を計算する。

    Arguments:
        w : 各ベクトルごとの重み係数。 サイズは [2]
        p : 球面上の荷重平均を取る球面上のベクトルを並べた行列。 サイズは [2, M]
    Returns:
        q : 球面上の荷重平均をとった値。サイズは [M]
        v : q = v @ p を満たす重み係数。サイズは [2]
    """
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
    """ Spherical Weighted Averae により、N 個の M 次元ベクトルについての球面線形補間を計算する。
        こちらはの元論文(https://mathweb.ucsd.edu/~sbuss/ResearchWeb/spheremean/paper.pdf)の
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
    """ Spherical Weighted Averae により、N 個の M 次元ベクトルについての球面線形補間を計算する。
        こちらは Riemann 多様体上の最急降下法で実装したもの。

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
        w_inv_sinc_th = w * inv_sinc_th

        # v @ p = q となる重み係数 v を算出。
        v = w_inv_sinc_th / np.sum(cos_th * w_inv_sinc_th)
        
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
    """ Spherical Weighted Averae により、N 個の M 次元ベクトルについての球面線形補間を計算する。
        こちらはRiemann 多様体上の L-BFGS 法で実装したもの。

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
    s = np.zeros( [max_mem, M ], dtype=w.dtype )
    y = np.zeros( [max_mem, M ], dtype=w.dtype )
    rho = np.zeros( [max_mem], dtype=w.dtype )
    alpha = np.zeros([max_mem], dtype=w.dtype )
    idx = 0

    cos_th = p @ q
    th =  np.arccos( cos_th )
    inv_sinc_th = 1.0/ np.sinc(th / np.pi)
    w_inv_sinc_th = w * inv_sinc_th

    # 勾配を算出
    grad = - ( w * 2 * th / np.sqrt( 1 - cos_th * cos_th ) ) @ p
    gamma = 1.0

    for l in range(max_loop):

        n_mem = min(l,max_mem)

        # v @ p = q となる重み係数 v を算出。
        v = w_inv_sinc_th / np.sum( cos_th * w_inv_sinc_th )

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
        inv_sinc_th = 1.0/np.sinc(th / np.pi)
        w_inv_sinc_th = w * inv_sinc_th

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
    
    # 2点間の Spherical Weighted Average が Slerp と一致するかのチェック
    print( "comparing 2-pint spherical average (Algorithm A1) to Slerp.")

    M = 256 # ベクトルの次元数

    diff_q = []
    diff_v = []
    for _ in range(1024):
        w = np.random.rand( 2 )
        w /= np.sum( w )

        p = np.random.randn( 2, M )
        p = p / np.linalg.norm( p, axis=1 )[:,np.newaxis]

        q_slerp, v_slerp = slerp(w, p)
        q_sph_avg, v_sph_avg, losses_sph_avg = sph_avg_a1( w, p )
        diff_q.append(np.max(np.abs( q_sph_avg - q_slerp )))
        diff_v.append(np.max(np.abs( v_slerp - v_sph_avg )))

    print( "\t max difference of q : {0:e}".format( np.max( diff_q ) ) )
    print( "\t max difference of v : {0:e}".format( np.max( diff_v ) ) )
    print( "")

    # %%

    # Spherical Weighted Average の各アルゴリズムの平均実行時間と計算結果の最大誤差をチェック

    print( "comparing each algorithms of spherical average.")

    N = 256 # ベクトルの数
    M = 256 # ベクトルの次元数
    L = 1000 # 平均を取る試行回数

    # beatrice-vst 内部では float で計算しているので、vst上での実行時間の推定はこっち
    dtype = np.float32
    # dtype = np.float64

    print( "( using {0:d}-bit float as dtype )".format(np.finfo(dtype).bits))

    eps = np.finfo(dtype).eps

    elapsed_time_A1 = 0
    elapsed_time_gd = 0
    elapsed_time_lbfgs = 0

    sum_count_A1 = 0
    sum_count_gd = 0
    sum_count_lbfgs = 0

    diff_q_gd = []
    diff_v_gd = []
    diff_q_lbfgs = []
    diff_v_lbfgs = []

    for l in range(L):
        w = np.random.rand( N ).astype(dtype)
        w /= np.sum( w )

        p = np.random.randn( N, M ).astype(dtype)
        p = p / np.linalg.norm( p, axis=1 )[:,np.newaxis]


        start_time = time.perf_counter()
        q_a1, v_a1, count_a1 = sph_avg_a1( w, p, eps=eps )
        elapsed_time_A1 += time.perf_counter() - start_time
        sum_count_A1 += count_a1

        start_time = time.perf_counter()
        q_gd, v_gd, count_gd = sph_avg_gradient_descent( w, p, eps=eps )
        elapsed_time_gd += time.perf_counter() - start_time
        sum_count_gd += count_gd

        start_time = time.perf_counter()
        q_lbfgs, v_lbfgs, count_lbfgs = sph_avg_l_bfgs( w, p, eps=eps )
        elapsed_time_lbfgs += time.perf_counter() - start_time
        sum_count_lbfgs += count_lbfgs

        diff_q_gd.append(np.max(np.abs(( q_a1 - q_gd ))))
        diff_v_gd.append(np.max(np.abs( v_a1 - v_gd )))
        diff_q_lbfgs.append(np.max(np.abs( q_a1 - q_lbfgs )))
        diff_v_lbfgs.append(np.max(np.abs( v_a1 - v_lbfgs )))

    # print( (q, v, loss) )
    print( "\t Original Algorithm A1")
    print( "\t\t average loop count : {0:.2f}".format( sum_count_A1 / L ) )
    print( "\t\t average elapsed time : {0:.2f} ms".format( (elapsed_time_A1 / L)*1000 ) )

    print( "\t Gradient descent")
    print( "\t\t average loop count : {0:.2f}".format( sum_count_gd / L ) )
    print( "\t\t average elapsed time : {0:.2f} ms".format( (elapsed_time_gd / L)*1000 ) )
    print( "\t\t max difference of q : {0:e}".format(np.max(diff_q_gd)))
    print( "\t\t max difference of v : {0:e}".format(np.max(diff_v_gd)))

    print( "\t L-BFGS")
    print( "\t\t average loop count : {0:.2f}".format( sum_count_lbfgs / L ) )
    print( "\t\t average elapsed time : {0:.2f} ms".format( (elapsed_time_lbfgs / L)*1000 ) )
    print( "\t\t max difference of q : {0:e}".format(np.max(diff_q_lbfgs)))
    print( "\t\t max difference of v : {0:e}".format(np.max(diff_v_lbfgs)))
