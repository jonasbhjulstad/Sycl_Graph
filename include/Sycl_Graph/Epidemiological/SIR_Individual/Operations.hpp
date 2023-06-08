#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Individual/Types.hpp>
import Operations.Types;
import Operations.Vertex;
import Operations.Edge;
import Base.Graph.Types;

template <Vertex_Buffer_type Vertex_Buffer_t> export struct SIR_Individual_Recovery_Op
    : public Vertex_Extract_Operation<Vertex_Buffer_t,
                                      SIR_Individual_Recovery_Op<Vertex_Buffer_t>> {
  typedef SIR_Individual_State_t Target_t;
  typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;

  float p_R = 0.0f;
  SIR_Individual_Recovery_Op(const Vertex_Buffer_t& buf, float p_R = 0.0f) : p_R(p_R) {}

  void invoke(const auto& v_acc, auto& result_acc, sycl::handler& h) const {
    const float p_R = this->p_R;
    h.parallel_for(v_acc.size(), [=](sycl::id<1> id) {
      Static_RNG::default_rng rng(id);
      Static_RNG::bernoulli_distribution<float> dist(p_R);
      result_acc[id] = dist(rng) ? SIR_INDIVIDUAL_R : v_acc.data[id];
    });
  }
  template <typename Graph_t> size_t target_buffer_size(const Graph_t& G) const {
    return G.template current_size<typename Vertex_Buffer_t::Vertex_t>();
  }
};

// Individual Infection Op: Chained with Individual Recovery Op as an inplace operation
template <Sycl::Edge_Buffer_type Edge_Buffer_t, Sycl::Vertex_Buffer_type Vertex_Buffer_t>
export struct SIR_Individual_Infection_Op
    : public Edge_Transform_Operation<Edge_Buffer_t,
                                      SIR_Individual_Infection_Op<Edge_Buffer_t, Vertex_Buffer_t>> {
  using Base_t
      = Edge_Transform_Operation<Edge_Buffer_t,
                                 SIR_Individual_Infection_Op<Edge_Buffer_t, Vertex_Buffer_t>>;
  static constexpr sycl::access::mode target_access_mode = sycl::access::mode::atomic;
  typedef SIR_Individual_State_t Target_t;
  typedef SIR_Individual_State_t Source_t;
  typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
  sycl::buffer<uint32_t>& seeds;
  float p_I = 0.0f;
  SIR_Individual_Infection_Op(const Edge_Buffer_t& edge_buf, Vertex_Buffer_t&,
                              sycl::buffer<uint32_t>& seeds, float p_I = 0.0f)
      : seeds(seeds), p_I(p_I) {
    assert(seeds.size() >= edge_buf.current_size() && "seeds buffer too small");
  }

  void invoke(const auto& edge_acc, const auto& from_acc, const auto& to_acc, auto& result_acc,
              sycl::handler& h) const {
    h.parallel_for(edge_acc.size(), [=](sycl::id<1> id) {
      auto id_from = edge_acc[id].from;
      auto id_to = edge_acc[id].to;
      auto seed_acc = seeds.get_access<sycl::access::mode::read_write>();
      if (edge_acc[id].is_valid() && (from_acc[id_from] == SIR_INDIVIDUAL_I)
          && (to_acc[id_to] == SIR_INDIVIDUAL_S)) {
        Static_RNG::default_rng rng(seed_acc[id]);
        seed_acc[id]++;
        auto p_I = edge_acc[id].data.p_I;
        Static_RNG::bernoulli_distribution<float> dist(p_I);
        if (dist(rng)) {
          to_acc[id_to] = SIR_INDIVIDUAL_I;
        }
        seed_acc[id]++;
      }
    });
  }
  template <typename Graph_t> size_t target_buffer_size(const Graph_t& G) const {
    return G.template current_size<Vertex_t>();
  }

  template <typename Graph_t> size_t source_buffer_size(const Graph_t& G) const {
    return G.template current_size<Vertex_t>();
  }
};

// Individual Infection Op: Chained with Individual Recovery Op as an inplace operation
export struct SIR_Individual_Population_Count_Extract_Op
    : public Vertex_Extract_Operation<SIR_Individual_Vertex_Buffer_t,
                                      SIR_Individual_Population_Count_Extract_Op> {
  using Base_t = Vertex_Extract_Operation<SIR_Individual_Vertex_Buffer_t,
                                          SIR_Individual_Population_Count_Extract_Op>;
  static constexpr sycl::access::mode target_access_mode = sycl::access::mode::write;
  typedef uint32_t Target_t;

  void invoke(const auto& v_acc, auto& target_acc, sycl::handler& h) const {
    h.single_task([=]() {
      target_acc[0] = 0;
      target_acc[1] = 0;
      target_acc[2] = 0;
      for (int i = 0; i < v_acc.size(); i++) {
        if (v_acc.data[i] == SIR_INDIVIDUAL_S) {
          target_acc[0]++;
        } else if (v_acc.data[i] == SIR_INDIVIDUAL_I) {
          target_acc[1]++;
        } else if (v_acc.data[i] == SIR_INDIVIDUAL_R) {
          target_acc[2]++;
        }
      }
    });
  }
  template <typename Graph_t> size_t target_buffer_size(const Graph_t& G) const { return 3; }
};

export struct SIR_Individual_Population_Count_Transform_Op
    : public Transform_Operation<SIR_Individual_Population_Count_Transform_Op> {
  using Base_t = Transform_Operation<SIR_Individual_Population_Count_Transform_Op>;
  static constexpr sycl::access::mode target_access_mode = sycl::access::mode::write;
  typedef uint32_t Target_t;
  typedef SIR_Individual_State_t Source_t;

  void invoke(const auto& source_acc, auto& target_acc, sycl::handler& h) const {
    h.single_task([=]() {
      target_acc[0] = 0;
      target_acc[1] = 0;
      target_acc[2] = 0;
      for (auto i = 0; i < source_acc.size(); i++) {
        if (source_acc[i] == SIR_INDIVIDUAL_S) {
          target_acc[0]++;
        } else if (source_acc[i] == SIR_INDIVIDUAL_I) {
          target_acc[1]++;
        } else if (source_acc[i] == SIR_INDIVIDUAL_R) {
          target_acc[2]++;
        }
      }
    });
  }
  template <typename Graph_t> size_t target_buffer_size(const Graph_t& G) const { return 3; }
};