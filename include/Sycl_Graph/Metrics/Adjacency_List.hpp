#ifndef SYCL_GRAPH_METRICS_ADJACENCY_LIST_HPP
#define SYCL_GRAPH_METRICS_ADJACENCY_LIST_HPP
#include <Sycl_Graph/Utils/work_groups.hpp>
#include <Sycl_Graph/Common_Kernels/Merge_Vectors.hpp>
namespace Sycl_Graph::USM
{
    sycl::event get_adjacency_list(Edge_t* edges, uint32_t N_edges, uint32_t N_vertices, uint32_t N_neighbors_max, uint32_t* adjlist, uint32_t* N_neighbors, uint32_t* N_neighbors_tot, sycl::queue& q)
    {
        auto nd_range = get_nd_range(q, N_vertices);

        auto count_event = q.submit([&](sycl::handler& h)
        {
            h.parallel_for(nd_range, [=](sycl::nd_item<1> it)
            {
                auto v_idx = it.get_global_id(0);
                auto N_cur_neighbors = 0;
                auto v_offset = v_idx * N_neighbors_max;
                for(auto e_idx = 0; e_idx < N_edges; e_idx++)
                {
                    if (N_cur_neighbors >= N_neighbors_max)
                    {
                        break;
                    }
                    auto e = edges[e_idx];
                    if(e.from == v_idx)
                    {
                        adjlist[v_offset + N_cur_neighbors] = e.to;
                        N_cur_neighbors++;
                    }
                }
                N_neighbors[v_idx] = N_cur_neighbors;
            });
        });
        return q.submit(Merge_Vectors<uint32_t>(adjlist, N_neighbors_tot, N_vertices, N_neighbors));
    }


    std::vector<std::vector<uint32_t>> get_adjacency_list(Edge_t* edges, uint32_t N_edges, uint32_t N_vertices, uint32_t N_neighbors_max, uint32_t* adjlist, uint32_t * N_neighbors, sycl::queue& q)
    {
        auto N_neighbors_tot = sycl::malloc_shared<uint32_t>(1, q);
        get_adjacency_list(edges, N_edges, N_vertices, N_neighbors_max, adjlist, N_neighbors, N_neighbors_tot, q).wait();
        std::vector<uint32_t> adjlist_flat(N_neighbors_tot[0]);
        q.memcpy(adjlist_flat.data(), adjlist, N_neighbors_tot[0] * sizeof(uint32_t)).wait();
        std::vector<std::vector<uint32_t>> adjlist_vec(N_vertices);
    }
}


#endif
