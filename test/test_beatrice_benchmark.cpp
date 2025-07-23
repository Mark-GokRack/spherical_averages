#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "SphAvg.hpp"
#include "SphAvgGD.hpp"
#include "SphAvgLBFGS.hpp"
#include "SphAvgLBFGS_align.hpp"

using namespace std;

using T = float;

void save_vec(string filename, const vector<T>& v, size_t N, size_t M = 1) {
  ofstream ofs(filename);
  for (size_t n = 0; n < N; n++) {
    for (size_t m = 0; m < M; m++) {
      ofs << scientific << setprecision(15) << v[n * M + m];
      if (m + 1 < M) {
        ofs << ",";
      }
    }
    ofs << endl;
  }
}




int main(int argc, char** argv) {
  size_t N1 = 256;
  size_t M1 = 256;

  std::vector<T> w1(N1);
  std::vector<T> p1(N1 * M1);

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  uniform_real_distribution<> dist_w(0.0, 1.0);
  normal_distribution<> dist_p(0.0, 1.0);

  size_t L = 1024;

  cout << "calculate original algorithm" << endl;


  size_t loop_count = 0;
  double elapsed_time_org = 0.0;
  for( size_t l = 0; l < L; l++ ){
    for (size_t n = 0; n < N1; n++) {
      w1[n] = dist_w(engine);
    }
    for (size_t n = 0; n < N1; n++) {
      for (size_t m = 0; m < M1; m++) {
        p1[n * M1 + m] = dist_p(engine);
      }
    }
    SphericalAverage<T> sph_avg(N1, M1, p1.data());
    auto start_time = chrono::system_clock::now();
    sph_avg.set_weights(N1, w1.data());
    while (!sph_avg.update()) {
      loop_count++;
    }
    auto end_time = chrono::system_clock::now();
    elapsed_time_org += static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  }

  cout << "num loop : " << loop_count / L << endl;
  cout << "average elapsed time : " << elapsed_time_org / L << " ms"<< endl;

  
  cout << "calculate L-BFGS method (with SIMD)" << endl;


  size_t N2 = 16;
  size_t M2 = 128;
  size_t K = 384 + 512;

  std::vector<T> w2(N2);
  std::vector<T> p2(N2 * M1);
  std::vector<T> p3(N2 * M2);

  loop_count = 0;
  double elapsed_time_new = 0.0;
  for( size_t l = 0; l < L; l++ ){
    for (size_t n = 0; n < N2; n++) {
      w2[n] = dist_w(engine);
    }
    for (size_t n = 0; n < N2; n++) {
      for (size_t m = 0; m < M1; m++) {
        p2[n * M1 + m] = dist_p(engine);
      }
      for (size_t m = 0; m < M2; m++) {
        p3[n * M2 + m] = dist_p(engine);
      }
    }
    SphericalAverageLBFGS_align<T> sph_avg_LBGS_1 (N2, M1, p2.data(), 4);
    SphericalAverageLBFGS_align<T> sph_avg_LBGS_2 (N2, M2, p3.data(), 4);
    auto start_time = chrono::system_clock::now();
    sph_avg_LBGS_1.set_weights(N2, w2.data());
    while (!sph_avg_LBGS_1.update()) {
      loop_count++;
    }
    for (size_t k = 0; k < K; k++){
      sph_avg_LBGS_2.set_weights(N2, w2.data());
      while (!sph_avg_LBGS_2.update()) {
        loop_count++;
      }
    }
    auto end_time = chrono::system_clock::now();
    elapsed_time_new += static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  }
  cout << "num loop : " << loop_count / L<< endl;
  cout << "average elapsed time : " << elapsed_time_new / L << " ms\t( " << elapsed_time_org / elapsed_time_new << " times faster )"<< endl;

}