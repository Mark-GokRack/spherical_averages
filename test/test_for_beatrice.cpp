#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "spherical_average.h"
#include "spherical_average_lbfgs.h"

using namespace std;
using namespace beatrice::common;

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
  size_t N_lim = 128;
  size_t M = 256;

  std::vector<T> w(N, 0.0);
  std::vector<T> p(N * M, 0.0);

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  uniform_real_distribution<> dist_w(0.0, 1.0);
  normal_distribution<> dist_p(0.0, 1.0);

  for (size_t n = 0; n < N_lim; n++) {
    w[n] = dist_w(engine);
  }
  for (size_t n = 0; n < N; n++) {
    for (size_t m = 0; m < M; m++) {
      p[n * M + m] = dist_p(engine);
    }
  }
  save_vec("w.csv", w, N);
  save_vec("p.csv", p, N, M);

#ifdef _DEBUG
  size_t L = 1;
#else
  size_t L = 1024;
#endif

  cout << "calculate original algorithm" << endl;
  SphericalAverage<T> sph_avg(N, M, p.data());

  size_t loop_count = 0;
  auto start_time = chrono::system_clock::now();
  for (size_t l = 0; l < L; l++) {
    sph_avg.SetWeights(N, w.data());
    while (!sph_avg.Update()) {
      loop_count++;
    }
  }
  auto end_time = chrono::system_clock::now();

  std::vector<T> q_org(M);
  sph_avg.ApplyWeights(N, M, p.data(), q_org.data());

  auto elapsed_time_a1 = static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  cout << "num loop : " << loop_count / L << endl;
  cout << "elapsed time : " << elapsed_time_a1 / L << " ms" << endl;
  cout << "elapsed time per each iteration : " << elapsed_time_a1 / loop_count
       << " ms" << endl;

  cout << "calculate new algorithm" << endl;

  SphericalAverageLBFGS<T> sph_avg_LBFGS(N, M, p.data(), N_lim, 4);

  loop_count = 0;
  start_time = chrono::system_clock::now();
  for (size_t l = 0; l < L; l++) {
    sph_avg_LBFGS.SetWeights(N, w.data());
    while (!sph_avg_LBFGS.Update()) {
      loop_count++;
    }
  }
  end_time = chrono::system_clock::now();

  std::vector<T> q_lbfgs(M);
  sph_avg_LBFGS.GetResult(M, q_lbfgs.data());

  auto elapsed_time_lbfgs = static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  cout << "num loop : " << loop_count / L << endl;
  cout << "elapsed time : " << elapsed_time_lbfgs / L << " ms\t( "
       << elapsed_time_a1 / elapsed_time_lbfgs << " times faster )" << endl;
  cout << "elapsed time per each iteration : "
       << elapsed_time_lbfgs / loop_count << " ms" << endl;

  T s = 0.0;
  T d = 0.0;

  for (size_t i = 0; i < M; i++) {
    T tmp = q_org[i] - q_lbfgs[i];
    s += q_org[i] * q_org[i];
    d += tmp * tmp;
  }
  auto sdr = (T)10.0 * log10(d / s);
  if (std::isnan(sdr)) {
    cerr << "nan" << endl;
  }

  cout << "diff SDR ov q : " << sdr << endl;
}