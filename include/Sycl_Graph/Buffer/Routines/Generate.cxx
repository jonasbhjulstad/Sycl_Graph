module;
#include <Sycl_Graph/Common/common.hpp>
export module Sycl.Buffer.Generate;
import Sycl.Buffer.Add;
export auto generate_seed_buf(uint32_t seed, uint32_t N_seeds, sycl::queue &q) {
  std::mt19937 gen(seed);
  sycl::buffer<uint32_t> seeds(N_seeds);
  // generate random uint32_t numbers
  std::vector<uint32_t> seed_vec(N_seeds);
  std::generate(seed_vec.begin(), seed_vec.end(), gen);
  buffer_add(seeds, seed_vec, q);
  return seeds;
}