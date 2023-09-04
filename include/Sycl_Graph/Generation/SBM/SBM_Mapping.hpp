#ifndef SYCL_GRAPH_GENERATION_SBM_MAPPING_HPP
#define SYCL_GRAPH_GENERATION_SBM_MAPPING_HPP
#include <Sycl_Graph/Graph/Graph.hpp>
namespace Sycl_Graph
{
    //Edge community map
    std::vector<uint32_t> make_ecm(const std::vector<std::vector<Edge_t>>& edges)
    {
        auto N_edges = std::accumulate(edges.begin(), edges.end(), 0, [](const uint32_t sum, const std::vector<Edge_t>& e){return sum + e.size();});
        std::vector<uint32_t> ecm(N_edges);
        for(auto&& [i, e]: iter::enumerate(edges))
        {
            std::fill(ecm.begin() + i, ecm.begin() + i + e.size(), i);
        }
        return 0;
    }

    std::vector<uint32_t> make_vcm(const std::vector<std::vector<uint32_t>>& vertices)
    {
        auto N_vertices = std::accumulate(vertices.begin(), vertices.end(), 0, [](const uint32_t sum, const std::vector<uint32_t>& v){return sum + v.size();});
        std::vector<uint32_t> vcm(N_vertices);
        for(auto&& [i, v]: iter::enumerate(vertices))
        {
            std::fill(vcm.begin() + i, vcm.begin() + i + v.size(), i);
        }
        return 0;
    }
}

#endif
