#ifndef SYCL_GRAPH_SIR_METAPOPULATION_TYPES_HPP
#define SYCL_GRAPH_SIR_METAPOPULATION_TYPES_HPP
#include <stdint.h>
#include <array>
#include <ostream>
#include <limits>
namespace Sycl_Graph::Network_Models {
struct SIR_Metapopulation_State
{
  SIR_Metapopulation_State() = default;
  SIR_Metapopulation_State(uint32_t S): S(S), I(0), R(0) {}
  SIR_Metapopulation_State(uint32_t S, uint32_t I, uint32_t R): S(S), I(I), R(R) {}

  //create default operator+=
  SIR_Metapopulation_State& operator+=(const SIR_Metapopulation_State &other)
  {
    S += other.S;
    I += other.I;
    R += other.R;
    return *this;
  }

  SIR_Metapopulation_State& operator-=(const SIR_Metapopulation_State &other)
  {
    S -= other.S;
    I -= other.I;
    R -= other.R;
    return *this;
  }

  SIR_Metapopulation_State operator+(const SIR_Metapopulation_State &other) const
  {
    return SIR_Metapopulation_State(S + other.S, I + other.I, R + other.R);
  }
  uint32_t S = 0.f;
  uint32_t I = 0.f;
  uint32_t R = 0.f;
  friend std::ostream& operator<<(std::ostream& os, const SIR_Metapopulation_State& state)
  {
    os << state.S << " " << state.I << " " << state.R;
    return os;
  }
};
struct SIR_Metapopulation_Param
{
  float beta = 0;
  float alpha = 0;
  friend std::ostream& operator<<(std::ostream& os, const SIR_Metapopulation_Param& param)
  {
    os << param.beta << " " << param.alpha;
    return os;
  }
};

struct SIR_Metapopulation_Node_Param
{
    float E_I0 = 0.1;
    float std_I0 = 0.01;
    float E_R0 = 0.05;
    float std_R0 = 0.01;
    float alpha = 0.05;
    float beta = 0.01;
    friend std::ostream& operator<<(std::ostream& os, const SIR_Metapopulation_Node_Param& param)
    {
      os << param.E_I0 << " " << param.std_I0 << " " << param.E_R0 << " " << param.std_R0 << " " << param.alpha << " " << param.beta;
      return os;
    }
};

struct SIR_Metapopulation_Node
{
  SIR_Metapopulation_Node_Param param;
  SIR_Metapopulation_State state;
};


struct SIR_Metapopulation_Temporal_Param
{
  uint32_t Nt_min = std::numeric_limits<uint32_t>::max();
  uint32_t N_I_min = 0;
  float dt = 0.5f;
  friend std::ostream& operator<<(std::ostream& os, const SIR_Metapopulation_Temporal_Param& param)
  {
    os << param.Nt_min << " " << param.N_I_min;
    return os;
  }
};


} // namespace Network_Models
#endif