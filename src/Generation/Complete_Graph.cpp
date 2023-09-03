#include <Sycl_Graph/Generation/Complete_Graph.hpp>
#include <execution>
#include <itertools.hpp>

std::vector<Edge_t> complete_graph(std::size_t N_vertices, bool directed, bool self_loops)
{
    std::size_t N_edges = (N_vertices * (N_vertices - 1) / 2) * (directed ? 1 : 2) - (self_loops ? 0 : N_vertices);
    std::vector<Edge_t> edges(N_edges);
    if (edges.max_size() < N_edges)
    {
        throw std::runtime_error("Too many edges requested");
    }

    if (directed)
    {
        std::vector<uint32_t> idx_0(N_vertices);
        std::vector<uint32_t> idx_1(N_vertices);
        std::iota(idx_0.begin(), idx_0.end(), 0);
        std::iota(idx_1.begin(), idx_1.end(), 0);
        std::transform(std::execution::par_unseq, idx_0.begin(), idx_0.end(), idx_1.begin(), edges.begin(), [](uint32_t i, uint32_t j)
                       { return Edge_t(i, j); });
    }
    else
    {
        for (int idx_0 = 0; idx_0 < N_vertices; idx_0++)
        {
            for (int idx_1 = idx_0 + 1; idx_1 < N_vertices; idx_1++)
            {
                edges.push_back(Edge_t(idx_0, idx_1));
            }
        }
    }
    if (!self_loops)
    {
        edges.erase(std::remove_if(edges.begin(), edges.end(), [](Edge_t e)
                                   { return e.first == e.second; }),
                    edges.end());
    }
    return edges;
}
