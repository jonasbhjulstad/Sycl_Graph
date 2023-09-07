#ifndef SYCL_GRAPH_GRAPH_ADJACENCY_LIST_HPP
#define SYCL_GRAPH_GRAPH_ADJACENCY_LIST_HPP
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
namespace Sycl_Graph::USM
{
    struct Adjacency_List_t
    {
        uint32_t N_vertices;
        uint32_t* neighbor_offsets;
        uint32_t* neighbors;
    };

    Adjacency_List_t initialize_adjacency_list(const std::vector<std::vector<uint32_t>>& neighbors, sycl::queue& q, std::vector<sycl::event>& res_events)
    {
        std::vector<uint32_t> neighbor_offsets(neighbors.size());
        std::vector<uint32_t> neighbors_flat;
        uint32_t offset = 0;
        for (auto&& n : neighbors)
        {
            neighbor_offsets[offset] = neighbors_flat.size();
            neighbors_flat.insert(neighbors_flat.end(), n.begin(), n.end());
            offset++;
        }
        Adjacency_List_t adj_list;
        res_events.resize(2);
        adj_list.neighbor_offsets = initialize_device_usm(neighbor_offsets, q, init_events[0]);
        adj_list.neighbors = initialize_device_usm(neighbors_flat, q, init_events[1]);
        return adj_list;
    }
}

#endif
