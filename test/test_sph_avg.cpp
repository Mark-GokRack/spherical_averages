#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "SphAvg.hpp"

using namespace std;

using T = double;

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

  SphericalAverage<T> sph_avg(N, M, p.data());

  sph_avg.set_weights(N, w.data());

  size_t loop_count = 0;
  auto start_time = chrono::system_clock::now();
  while (!sph_avg.update()) {
    loop_count++;
  }
  auto end_time = chrono::system_clock::now();

  auto elapsed_time = static_cast<T>(
      chrono::duration_cast<chrono::microseconds>(end_time - start_time)
          .count() /
      1000.0);
  cout << "num loop : " << loop_count << endl;
  cout << "elapsed time : " << elapsed_time << endl;

  save_vec("v.csv", sph_avg.get_weights(), N);
}