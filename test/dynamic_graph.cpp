#include <Sycl_Graph/Graph/Dynamic/Graph.hpp>


int main()
{
    using namespace Sycl_Graph::Dynamic;

    using Vertex_t = Vertex<int, int>;
    using Edge_t = Edge<int, int>;
    Graph<Vertex_t, Edge_t> G;
    std::vector<Vertex_t> vertices;
    vertices.push_back(Vertex_t(0, 0));
    vertices.push_back(Vertex_t(1, 1));
    vertices.push_back(Vertex_t(2, 2));
    vertices.push_back(Vertex_t(3, 3));

    G.add_vertices(vertices);

    std::vector<Edge_t> edges;
    edges.push_back(Edge_t(0, 1, 0));
    edges.push_back(Edge_t(1, 2, 1));
    edges.push_back(Edge_t(2, 3, 2));
    edges.push_back(Edge_t(3, 0, 3));

    G.add_edges(edges);

}