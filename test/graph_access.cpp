#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Buffer/Sycl/Invariant/Vertex_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Invariant/Edge_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Invariant/Graph.hpp>
#include <Sycl_Graph/Buffer/Invariant/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Invariant/Vertex_Buffer.hpp>

using namespace Sycl_Graph;
typedef Sycl_Graph::Base::Vertex<float> fVertex;
typedef Sycl_Graph::Base::Vertex<int> iVertex;
typedef Sycl_Graph::Base::Edge<float> fEdge;
typedef Sycl_Graph::Invariant::Edge<fEdge, iVertex, fVertex> i_f_edges;
typedef Sycl_Graph::Invariant::Edge<fEdge, fVertex, iVertex> f_i_edges;


int main()
{
    sycl::queue q(sycl::gpu_selector_v);

    std::vector<fVertex> fvertices = {fVertex{0, 0.0f}, fVertex{1, 1.0f}, fVertex{2, 2.0f}, fVertex{3, 3.0f}};
    std::vector<iVertex> ivertices = {iVertex{0, 0}, iVertex{1, 1}, iVertex{2, 2}, iVertex{3, 3}};

    std::vector<fEdge> i_f_edges = {fEdge{0, 1}, fEdge{1, 2}, fEdge{2, 3}, fEdge{3, 2}};
    std::vector<fEdge> f_i_edges = {fEdge{0, 1}, fEdge{1, 2}, fEdge{2, 3}, fEdge{3, 2}};

    Sycl_Graph::Sycl::Base::Vertex_Buffer fv_buf(q, fvertices);
    Sycl_Graph::Sycl::Base::Vertex_Buffer iv_buf(q, ivertices);
    Sycl_Graph::Sycl::Base::Edge_Buffer i_f_e_buf(q, i_f_edges);
    Sycl_Graph::Sycl::Base::Edge_Buffer f_i_e_buf(q, f_i_edges);

    Sycl_Graph::Sycl::Invariant::Vertex_Buffer vertex_buffer(fv_buf, iv_buf);
    Sycl_Graph::Sycl::Invariant::Edge_Buffer edge_buffer(i_f_e_buf, f_i_e_buf);
    Sycl_Graph::Sycl::Invariant::Graph graph(vertex_buffer, edge_buffer, q);

    std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges() << " edges." << std::endl;

    auto i_f_acc = graph.template get_access<()

    return 0;
}