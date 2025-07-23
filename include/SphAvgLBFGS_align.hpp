#pragma once

#define _USE_MATH_DEFINES 
#include <cmath>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <limits>
#include <cassert>
#include "AlignedVector.hpp"
#include <numeric>

/**
 * This class implements L-BFGS method for spherical averages
 * to calculate spherical linear interpolation between two or more vectors.
 * see also :  Spherical Averages (https://mathweb.ucsd.edu/~sbuss/ResearchWeb/spheremean/index.html)
 */

template<typename T>
class SphericalAverageLBFGS_align{
public:

    SphericalAverageLBFGS_align( )
        : N( 0 ), M( 0 ), K( 0 ), converged(false)
        , w(), p(), q(), v(), g()
        , mem_idx(0), gamma(0), d(), s(), t(), r(), a()
    {}

    SphericalAverageLBFGS_align( size_t num_point, size_t num_feature, T* unnormalized_vectors, size_t num_memory = 8 )
        : N( num_point ), M( num_feature ), K( num_memory ), converged(false)
        , w(), p(), q(), v(), g()
        , mem_idx(0), gamma(0), d(), s(), t(), r(), a()
    {
        initialize( N, M, unnormalized_vectors, K );
    }
    ~SphericalAverageLBFGS_align() = default;

    void initialize( size_t num_point, size_t num_feature, T* unnormalized_vectors, size_t num_memory = 8 ){
        N = num_point;
        M = num_feature;
        K = num_memory;

        w.resize(N, (T)0.0);   // size = N
        p.resize(N*M, (T)0.0); // size = N * M
        q.resize(M, (T)0.0);   // size = M
    
        v.resize(N, (T)0.0);   // size = N
        g.resize(M, (T)0.0);   // size = M
        d.resize(M, (T)0.0);   // size = M

        s.resize(K*M, (T)0.0);   // size = M
        t.resize(K*M, (T)0.0);   // size = M
        r.resize(K, (T)0.0);   // size = M
        a.resize(K, (T)0.0);   // size = M

        std::copy_n(
            unnormalized_vectors, N * M, p.begin()
        );
        for (size_t n = 0; n < N; n++){
            normalize_vector( M, &p[ n * M ]);
        }
    }

    void set_weights( size_t num_point, T* weights ){
        assert( N == num_point );
        converged = false;
        std::copy_n( weights, N, w.begin() );
        if( normalize_weight( N, w.data() ) ){
            weighted_sum( w.data(), p.data(), q.data() );
            if( !normalize_vector( M, q.data() ) ){
                converged = true;
            }
        }else{
            converged = true;
        }
        if( converged ){
            std::fill_n( v.begin(), N, (T)0.0);
        }else{

            mem_idx = 0;
            gamma = 1.0;
            std::fill_n( s.begin(), K*M, (T)0.0);
            std::fill_n( t.begin(), K*M, (T)0.0);
            std::fill_n( r.begin(), K, (T)0.0);
            std::fill_n( a.begin(), K, (T)0.0);
            std::fill_n( g.begin(), M, (T)0.0);

            update_v_g_d();
        }
    }

    bool update( void ){
        if( converged ){
            return true;
        }
        T norm_d = sqrt( dot( M, d.data(), d.data() ) );
        if( 
            norm_d >= 8*std::numeric_limits<T>::epsilon()
        ){
            update_q_s();
            update_v_g_d_t();
            update_gamma_r();
        }else{
            converged = true;
        }
        return converged;
    }

    const AlignedVector<T, 64> get_weights( void ){
        return v;
    }

private:

    inline T dot( size_t len, const T* __restrict x1, T* __restrict x2){
        const T* xx1 = std::assume_aligned<64>( x1 );
        const T* xx2 = std::assume_aligned<64>( x2 );
        T y = (T)0;
        #pragma clang loop vectorize(enable) unroll(enable)
        for( size_t l = 0; l < len; l++ ){
            y += xx1[l] * xx2[l];
        }
        return y;
    }

    inline void mul_c( size_t len, T a,  T* x ){
        T* xx = std::assume_aligned<64>( x );
        #pragma clang loop vectorize(enable) unroll(enable)
        for( size_t l = 0; l < len; l++ ){
            xx[l] *= a;
        }
    }

    inline void add_product_c( size_t len, T a, const T* __restrict x, T* __restrict y ){
        const T* xx = std::assume_aligned<64>( x );
        T* yy = std::assume_aligned<64>( y );
        #pragma clang loop vectorize(enable) unroll(enable)
        for( size_t l = 0; l < len; l++ ){
            yy[l] += a * xx[l];
        }
    }

    inline T sum( size_t len, const T* __restrict x){
        const T* xx = std::assume_aligned<64>( x );
        T y = 0;
        #pragma clang loop vectorize(enable) unroll(enable)
        for( size_t l = 0; l < len; l++ ){
            y += xx[l];
        }
        return y;
    }


    inline bool normalize_vector( size_t len, T* x ){
        const T* xx = std::assume_aligned<64>( x );
        T norm = sqrt( dot( len, x, x ) );
        if( norm > 0.0 ){
            T scale_factor = ((T)1.0) / norm;
            mul_c( len, scale_factor, x );
            return true;
        }else{
            //std::fill_n( x, len, (T)0.0 );
            return false;
        }
    }

    inline void project_vector_to_plane( size_t len, const T * __restrict x, T* __restrict y ){
        T minus_inner_product = -dot( len, x, y );
        add_product_c( len, minus_inner_product, x, y );
    }

    inline bool normalize_weight( size_t len, T* __restrict x ){
        T sum_x = sum( len, x );
        for( size_t l = 0; l < len; l++ ){
            sum_x += x[l];
        }
        if( sum_x > 0.0 ){
            T scale_factor = ((T)1.0) / sum_x;
            mul_c( len, scale_factor, x );
            return true;
        }else{
            //std::fill_n( x, len, (T)0.0 );
            return false;
        }
    }

    void weighted_sum( const T* weights, const T* x, T* y ){
        std::fill_n( y, M, (T)0.0 );
        for( size_t n = 0; n < N; n++ ){
            add_product_c( M, weights[n], &x[n*M], y );
        }
    }

    T sinc( T x ){
        static const T threshold_0 = std::numeric_limits<T>::epsilon();
        static const T threshold_1 = sqrt( threshold_0 );
        static const T threshold_2 = sqrt( threshold_1 );
        T y = (T)0.0;
        T abs_x = std::abs( x );
        if( abs_x >= threshold_2 ){
            y = sin( x ) / x;
        }else{
            y = (T)1.0;
            if( abs_x >= threshold_0 ){
                T x2 = x * x;
                y -= x2 / ((T)6.0);
                if( abs_x >= threshold_1 ){
                    y += x2 * x2 / ((T)120.0);
                }
            }
        }
        return y;
    }

    void update_v_g_d( void ){
        T sum_w_c_s = (T)0.0;
        std::fill_n(g.begin(), M, (T)0.0);

        for( size_t n = 0; n < N; n++ ){
            T cos_th = dot( M, &p[n*M], q.data());
            T theta = acos( cos_th );
            T inv_sinc_th = ((T)1.0) / ( sinc( theta ) + std::numeric_limits<T>::epsilon() );
            sum_w_c_s += w[n] * cos_th * inv_sinc_th;

            v[n] = w[n] * inv_sinc_th;

            T a_n = - ((T)2.0) * w[ n ] * theta / sqrt( ((T)1.0) - cos_th * cos_th );
            add_product_c( M, a_n, &p[n*M], g.data());
        }

        T inv_sum_w_c_s = ((T)1.0) / ( sum_w_c_s + std::numeric_limits<T>::epsilon() );
        mul_c( N, inv_sum_w_c_s, v.data() );

        project_vector_to_plane( M, q.data(), g.data() );

        std::copy( g.begin(), g.end(), d.begin() );
        for( size_t k = 0; k < K; k++ ){
            size_t idx = ( mem_idx - k - 1 + K ) % K;
            a[idx] = r[idx] * dot( M, &s[idx*M], d.data() );
            add_product_c( M, -a[idx], &t[idx*M], d.data() );
        }
        mul_c( M, gamma, d.data());
        for( size_t k = 0; k < K; k++ ){
            size_t idx = ( mem_idx + k ) % K;
            T b = r[idx] * dot( M, &t[idx*M], d.data() );
            add_product_c( M,  (a[idx]-b), &s[idx*M], d.data() );
        }

    }

    void update_v_g_d_t( void ){

        std::copy( g.begin(), g.end(), &t[mem_idx*M] );

        update_v_g_d();

        T* __restrict tt = std::assume_aligned<64>(&t[mem_idx*M]);
        const T* __restrict gg = std::assume_aligned<64>(g.data()); 
        #pragma clang loop vectorize(enable) unroll(enable)
        for (size_t m = 0; m < M; m++){
            tt[ m ] = gg[ m ] - tt[ m ];
        }
        project_vector_to_plane( M, q.data(), &t[mem_idx*M] );
    }

    void update_q_s( void ){

        std::copy( q.begin(), q.end(), &s[mem_idx*M] );

        T* __restrict qq = std::assume_aligned<64>(q.data());
        const T* __restrict dd = std::assume_aligned<64>(d.data()); 
        #pragma clang loop vectorize(enable) unroll(enable)
        for (size_t m = 0; m < M; m++){
            qq[ m ] -= dd[ m ];
        }
        normalize_vector( M, q.data() );

        T* __restrict ss = std::assume_aligned<64>(&s[mem_idx*M]);
        #pragma clang loop vectorize(enable) unroll(enable)
        for (size_t m = 0; m < M; m++){
            ss[ m ] = qq[ m ] - ss[ m ];
        }
    }

    void update_gamma_r( void ){
        gamma = dot( M, &s[mem_idx*M], &t[mem_idx*M] );
        r[mem_idx] = ((T)1.0) / gamma;
        gamma /= dot( M, &t[mem_idx*M], &t[mem_idx*M] );
        mem_idx +=1;
        if( mem_idx >= K ){
            mem_idx = 0;
        }

    }

    size_t N;
    size_t M;
    size_t K;

    bool converged;

    // vectors in original space
    AlignedVector<T, 64> w;   // size = N
    AlignedVector<T, 64> p;   // size = N * M

    AlignedVector<T, 64> q;   // size = M
    
    AlignedVector<T, 64> v;   // size = N
    AlignedVector<T, 64> g;   // size = M

    size_t mem_idx;
    T gamma;
    AlignedVector<T, 64> d;   // size = M
    AlignedVector<T, 64> s;   // size = K * M 
    AlignedVector<T, 64> t;   // size = K * M
    AlignedVector<T, 64> r;   // size = K
    AlignedVector<T, 64> a;   // size = K 

};