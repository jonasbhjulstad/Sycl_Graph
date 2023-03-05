#ifndef SYCL_GRAPH_SIR_BERNOULLI_TYPES_HPP
#define SYCL_GRAPH_SIR_BERNOULLI_TYPES_HPP
#include "sycl/aliases.hpp"
#include <stdint.h>
#include <CL/sycl.hpp>
namespace Sycl_Graph::Network_Models {
enum SIR_Individual_State: int { SIR_INDIVIDUAL_S = 0, SIR_INDIVIDUAL_I = 1, SIR_INDIVIDUAL_R = 2 };
struct SIR_Edge {};

template <typename dType = float> struct SIR_Bernoulli_Temporal_Param {
  dType p_I = 0.05;
  dType p_R = 0.01;
  uint32_t Nt_min = std::numeric_limits<uint32_t>::max();
  uint32_t N_I_min = 0;
};
} // namespace Network_Models
#endif