#ifndef SYCL_GRAPH_COMMON_KERNELS_MERGE_VECTORS_HPP
#define SYCL_GRAPH_COMMON_KERNELS_MERGE_VECTORS_HPP

namespace Sycl_Graph
{
template <typename T>
struct Merge_Vectors {
    Merge_Vectors(T* p_data, uint32_t* N_tot, uint32_t N_global, uint32_t* N_per_thread)
        : e_acc(p_edges), N_global(N_global), N_tot(N_tot), N_per_thread(N_per_thread) {}
    void operator()() const {
      auto N_merged = 0;
      auto offset = 0;
      for (int i = 0; i < N_global; i++) {
        for (int j = 0; j < N_per_thread[i]; j++) {
          p_data[N_merged] = p_data[offset + j];
          N_merged_edges++;
        }
        offset += N_per_thread[i];
      }
      N_tot[0] = N_merged_edges;
    }

    const uint32_t N_global;

  private:
    T* p_data;
    uint32_t* N_tot;
    uint32_t* N_per_thread;
  };

}  // namespace Sycl_Graph


#endif
