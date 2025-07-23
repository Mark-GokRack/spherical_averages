#pragma once

#define _USE_MATH_DEFINES 
#include <cmath>
#include <cstddef>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cassert>

/**
 * This class implements gradient descent method for spherical averages
 * to calculate spherical linear interpolation between two or more vectors.
 * see also :  Spherical Averages (https://mathweb.ucsd.edu/~sbuss/ResearchWeb/spheremean/index.html)
 */

template<typename T>
class SphericalAverageGD{
public:

    SphericalAverageGD( )
        : N( 0 ), M( 0 ), converged(false)
        , w(), p(), q(), v(), g()
    {}

    SphericalAverageGD( size_t num_point, size_t num_feature, T* unnormalized_vectors )
        : N( num_point ), M( num_feature ), converged(false)
        , w(), p(), q(), v(), g()
    {
        initialize( N, M, unnormalized_vectors );
    }
    ~SphericalAverageGD() = default;

    void initialize( size_t num_point, size_t num_feature, T* unnormalized_vectors ){
        N = num_point;
        M = num_feature;
        w.resize(N, (T)0.0);   // size = N
        p.resize(N*M, (T)0.0); // size = N * M
        q.resize(M, (T)0.0);   // size = M
    
        v.resize(N, (T)0.0);   // size = N
        g.resize(M, (T)0.0);   // size = M
        //u.resize(M, (T)0.0);   // size = M
        //loss = std::numeric_limits<T>::max();

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
            update_v_g();
        }
    }

    bool update( void ){
        if( converged ){
            return true;
        }
        T norm_g = sqrt( dot( M, g.data(), g.data() ) );
        /*
        weighted_sum( v.data(), p.data(), u.data() );
        for( size_t m = 0; m < M; m++ ){
            u[m] -= q[m];
        }
        T norm_u = sqrt( dot( M, u.data(), u.data() ) );
        */
        if( 
            norm_g >= 8*std::numeric_limits<T>::epsilon()
            // epsilon そのままだと結構止まらないことがあったので少し大きくしておく
            // && loss - norm_u >= std::numeric_limits<T>::epsilon() 
        ){
            //loss = norm_u;
            update_q();
            update_v_g();
        }else{
            converged = true;
        }
        return converged;
    }

    const std::vector<T>& get_weights( void ){
        return v;
    }

private:

    T dot( size_t len, const T* x1, const T* x2){
        T y = (T)0;
        for( size_t l = 0; l < len; l++ ){
            y += x1[l] * x2[l];
        }
        return y;
    }

    bool normalize_vector( size_t len, T* x ){
        T norm = sqrt( dot( len, x, x ) );
        if( norm > 0.0 ){
            T scale_factor = ((T)1.0) / norm;
            for( size_t l = 0; l < len; l++ ){
                x[l] *= scale_factor;
            }
            return true;
        }else{
            //std::fill_n( x, len, (T)0.0 );
            return false;
        }
    }

    void project_vector_to_plane( size_t len, const T* x, T* y ){
        T inner_product = dot( len, x, y );
        for( size_t l = 0; l < len; l++ ){
            y[l] -= inner_product * x[l];
        }
    }

    bool normalize_weight( size_t len, T* x ){
        T sum = 0.0;
        for( size_t l = 0; l < len; l++ ){
            sum += x[l];
        }
        if( sum > 0.0 ){
            T scale_factor = ((T)1.0) / sum;
            for( size_t l = 0; l < len; l++ ){
                x[l] *= scale_factor;
            }
            return true;
        }else{
            //std::fill_n( x, len, (T)0.0 );
            return false;
        }
    }

    void weighted_sum( const T* weights, const T* x, T* y ){
        std::fill_n( y, M, (T)0.0 );
        for( size_t n = 0; n < N; n++ ){
            for( size_t m = 0; m < M; m++ ){
                y[m] += weights[n] * x[ n * M + m ];
            }
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

    void update_v_g( void ){
        T sum_w_c_s = (T)0.0;
        
        for( size_t m = 0; m < M; m++ ){
            g[ m ] = (T)0.0;
        }

        for( size_t n = 0; n < N; n++ ){
            T cos_th = dot( M, &p[n*M], q.data());
            T theta = acos( cos_th );
            T inv_sinc_th = ((T)1.0) / ( sinc( theta ) + std::numeric_limits<T>::epsilon() );
            sum_w_c_s += w[n] * cos_th * inv_sinc_th;

            v[n] = w[n] * inv_sinc_th;

            T a_n = - ((T)2.0) * w[ n ] * theta / sqrt( ((T)1.0) - cos_th * cos_th );
            for( size_t m = 0; m < M; m++ ){
                g[ m ] += a_n * p[ n*M + m ];
            }
        }

        T inv_sum_w_c_s = ((T)1.0) / ( sum_w_c_s + std::numeric_limits<T>::epsilon() );
        for( size_t n = 0; n < N; n++ ){
            v[n] *= inv_sum_w_c_s;
        }

        project_vector_to_plane( M, q.data(), g.data() );

    }

    void update_q( void ){
        for (size_t m = 0; m < M; m++){
            q[ m ] -= g[ m ];
        }
        normalize_vector( M, q.data() );
    }

    size_t N;
    size_t M;
    bool converged;

    // vectors in original space
    std::vector<T> w;   // size = N
    std::vector<T> p;   // size = N * M

    std::vector<T> q;   // size = M
    
    std::vector<T> v;   // size = N
    std::vector<T> g;   // size = M

    // std::vector<T> u;   // size = M
    //T loss;

};