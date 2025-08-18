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
  size_t N = 256;
  size_t M = 256;

  std::vector<T> w(N);
  std::vector<T> p(N * M);

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  uniform_real_distribution<> dist_w(0.0, 1.0);
  normal_distribution<> dist_p(0.0, 1.0);

  for (size_t n = 0; n < N; n++) {
    w[n] = dist_w(engine);
  }
  for (size_t n = 0; n < N; n++) {
    for (size_t m = 0; m < M; m++) {
      p[n * M + m] = dist_p(engine);
    }
  }
  save_vec("w.csv", w, N);
  save_vec("p.csv", p, N, M);

  size_t L = 1024;

  cout << "calculate original algorithm" << endl;
  SphericalAverage<T> sph_avg(N, M, p.data());

  size_t loop_count = 0;
  auto start_time = chrono::system_clock::now();
  for (size_t l = 0; l < L; l++) {
    sph_avg.set_weights(N, w.data());
    while (!sph_avg.update()) {
      loop_count++;
    }
  }
  auto end_time = chrono::system_clock::now();

  auto& v_org = sph_avg.get_weights();

  auto elapsed_time_a1 = static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  cout << "num loop : " << loop_count / L << endl;
  cout << "elapsed time : " << elapsed_time_a1 / L << " ms" << endl;
  cout << "elapsed time per each iteration : " << elapsed_time_a1 / loop_count
       << " ms" << endl;

  cout << "calculate gradient descent method" << endl;

  SphericalAverageGD<T> sph_avg_gd(N, M, p.data());

  loop_count = 0;
  start_time = chrono::system_clock::now();
  for (size_t l = 0; l < L; l++) {
    sph_avg_gd.set_weights(N, w.data());
    while (!sph_avg_gd.update()) {
      loop_count++;
    }
  }
  end_time = chrono::system_clock::now();

  auto& v_gd = sph_avg_gd.get_weights();

  auto elapsed_time = static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  cout << "num loop : " << loop_count / L << endl;
  cout << "elapsed time : " << elapsed_time / L << " ms\t( "
       << elapsed_time_a1 / elapsed_time << " times faster )" << endl;
  cout << "elapsed time per each iteration : " << elapsed_time / loop_count
       << " ms" << endl;

  T s = (T)0.0;
  T d = (T)0.0;

  for (size_t n = 0; n < N; n++) {
    T tmp = v_org[n] - v_gd[n];
    s += v_org[n] * v_org[n];
    d += tmp * tmp;
  }
  T sdr = 10.0 * log10(d / s);
  if (std::isnan(sdr)) {
    cerr << "nan" << endl;
  }

  cout << "diff SDR ov v : " << sdr << endl;

  cout << "calculate L-BFGS method" << endl;

  SphericalAverageLBFGS<T> sph_avg_LBFGS(N, M, p.data(), 4);

  loop_count = 0;
  start_time = chrono::system_clock::now();
  for (size_t l = 0; l < L; l++) {
    sph_avg_LBFGS.set_weights(N, w.data());
    while (!sph_avg_LBFGS.update()) {
      loop_count++;
    }
  }
  end_time = chrono::system_clock::now();

  auto& v_lbfgs = sph_avg_LBFGS.get_weights();

  elapsed_time = static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  cout << "num loop : " << loop_count / L << endl;
  cout << "elapsed time : " << elapsed_time / L << " ms\t( "
       << elapsed_time_a1 / elapsed_time << " times faster )" << endl;
  cout << "elapsed time per each iteration : " << elapsed_time / loop_count
       << " ms" << endl;

  s = (T)0.0;
  d = (T)0.0;

  for (size_t n = 0; n < N; n++) {
    T tmp = v_org[n] - v_lbfgs[n];
    s += v_org[n] * v_org[n];
    d += tmp * tmp;
  }
  sdr = 10.0 * log10(d / s);
  if (std::isnan(sdr)) {
    cerr << "nan" << endl;
  }

  cout << "diff SDR ov v : " << sdr << endl;

  cout << "calculate L-BFGS method (with OpenMP SIMD)" << endl;

  SphericalAverageLBFGS_align<T> sph_avg_LBFGS_SIMD(N, M, p.data(), 4);

  loop_count = 0;
  start_time = chrono::system_clock::now();
  for (size_t l = 0; l < L; l++) {
    sph_avg_LBFGS_SIMD.set_weights(N, w.data());
    while (!sph_avg_LBFGS_SIMD.update()) {
      loop_count++;
    }
  }
  end_time = chrono::system_clock::now();

  auto& v_lbfgs_simd = sph_avg_LBFGS_SIMD.get_weights();

  elapsed_time = static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  cout << "num loop : " << loop_count / L << endl;
  cout << "elapsed time : " << elapsed_time / L << " ms\t( "
       << elapsed_time_a1 / elapsed_time << " times faster )" << endl;
  cout << "elapsed time per each iteration : " << elapsed_time / loop_count
       << " ms" << endl;

  s = (T)0.0;
  d = (T)0.0;

  for (size_t n = 0; n < N; n++) {
    T tmp = v_org[n] - v_lbfgs_simd[n];
    s += v_org[n] * v_org[n];
    d += tmp * tmp;
  }
  sdr = 10.0 * log10(d / s);
  if (std::isnan(sdr)) {
    cerr << "nan" << endl;
  }

  cout << "diff SDR ov v : " << sdr << endl;

  save_vec("v.csv", v_org, N);
}