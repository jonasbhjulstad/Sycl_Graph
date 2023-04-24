#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Sycl/Invariant_Graph.hpp>

using namespace Sycl_Graph;
typedef Sycl_Graph::Vertex<float> fVertex;
typedef Sycl_Graph::Vertex<int> iVertex;
typedef Sycl_Graph::Edge<float> fEdge;
typedef Sycl_Graph::Edge<fEdge, iVertex, fVertex> i_f_edges;
typedef Sycl_Graph::Edge<fEdge, fVertex, iVertex> f_i_edges;


int main()
{
    sycl::queue q(sycl::gpu_selector_v);

    std::vector<fVertex> fvertices = {fVertex{0, 0.0f}, fVertex{1, 1.0f}, fVertex{2, 2.0f}, fVertex{3, 3.0f}};
    std::vector<iVertex> ivertices = {iVertex{0, 0}, iVertex{1, 1}, iVertex{2, 2}, iVertex{3, 3}};

    std::vector<fEdge> i_f_edges = {fEdge{0, 1}, fEdge{1, 2}, fEdge{2, 3}, fEdge{3, 2}};
    std::vector<fEdge> f_i_edges = {fEdge{0, 1}, fEdge{1, 2}, fEdge{2, 3}, fEdge{3, 2}};

    Sycl_Graph::Sycl::Vertex_Buffer fv_buf(q, fvertices);
    Sycl_Graph::Sycl::Vertex_Buffer iv_buf(q, ivertices);
    Sycl_Graph::Sycl::Edge_Buffer i_f_e_buf(q, i_f_edges);
    Sycl_Graph::Sycl::Edge_Buffer f_i_e_buf(q, f_i_edges);

    Sycl_Graph::Sycl::Buffer_Pack vertex_buffer(fv_buf, iv_buf);
    Sycl_Graph::Sycl::Buffer_Pack edge_buffer(i_f_e_buf, f_i_e_buf);
    Sycl_Graph::Sycl::Graph graph(vertex_buffer, edge_buffer, q);

    std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges() << " edges." << std::endl;

    // auto& i_f_e_buf_ref = graph.edge_buf.get_edges<decltype(i_f_e_buf)>();
    // static_assert(std::is_same<decltype(i_f_e_buf_ref), decltype(i_f_e_buf)>::value);

    return 0;
}