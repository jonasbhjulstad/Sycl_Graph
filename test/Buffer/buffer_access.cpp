#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer_Pack.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer_Pack.hpp>
#include <Sycl_Graph/Graph/Sycl/Invariant_Graph.hpp>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer_Pack.hpp>
#include <Sycl_Graph/Buffer/Base/Vertex_Buffer_Pack.hpp>
#include <itertools.hpp>

using namespace Sycl_Graph;
typedef Sycl_Graph::Vertex<float> fVertex;
typedef Sycl_Graph::Vertex<int> iVertex;
typedef Sycl_Graph::Edge<float, iVertex, fVertex> i_f_edge_t;
typedef Sycl_Graph::Edge<float, fVertex, iVertex> f_i_edge_t;

int main()
{

    std::vector<uint32_t> e_idx(4);
    std::iota(e_idx.begin(), e_idx.end(), 0);
    sycl::queue q(sycl::gpu_selector_v);

    std::vector<fVertex> fvertices = {fVertex{0, 0.0f}, fVertex{1, 1.0f}, fVertex{2, 2.0f}, fVertex{3, 3.0f}};
    std::vector<iVertex> ivertices = {iVertex{0, 0}, iVertex{1, 1}, iVertex{2, 2}, iVertex{3, 3}};

    std::vector<i_f_edge_t> i_f_edges;
    std::vector<f_i_edge_t> f_i_edges;
    for(auto&& comb: iter::combinations(e_idx, 2))
    {
        i_f_edges.push_back(i_f_edge_t(comb[0], comb[1], 0.0f));
        f_i_edges.push_back(f_i_edge_t(comb[1], comb[0], 1.0f));
    }
    // std::vector<i_f_edge_t> i_f_edges = {fEdge{{0,1},0}, fEdge{1, 2}, fEdge{2, 3}, fEdge{3, 2}};
    // std::vector<f_i_edge_t> f_i_edges = {fEdge{0, 1}, fEdge{1, 2}, fEdge{2, 3}, fEdge{3, 2}};

    Sycl_Graph::Sycl::Vertex_Buffer fv_buf(q, fvertices);
    Sycl_Graph::Sycl::Vertex_Buffer iv_buf(q, ivertices);
    Sycl_Graph::Sycl::Edge_Buffer i_f_e_buf(q, i_f_edges);
    Sycl_Graph::Sycl::Edge_Buffer f_i_e_buf(q, f_i_edges);

    Sycl_Graph::Sycl::Vertex_Buffer_Pack vertex_buffer(fv_buf, iv_buf);
    Sycl_Graph::Sycl::Edge_Buffer_Pack edge_buffer(i_f_e_buf, f_i_e_buf);
    Sycl_Graph::Sycl::Graph graph(vertex_buffer, edge_buffer, q);
    q.submit([&](sycl::handler& h)
    {
        auto fv_acc = fv_buf.get_access<sycl::access::mode::read_write>(h);
        static_assert(std::is_same<decltype(fv_acc[0]), fVertex>::value);
        auto i_f_acc = i_f_e_buf.get_access<sycl::access::mode::read_write>(h);
        static_assert(std::is_same<decltype(i_f_acc[0]), i_f_edge_t>::value);

        auto g_fv_acc = graph.template get_access<sycl::access::mode::read_write, fVertex>(h);
        static_assert(std::is_same<decltype(g_fv_acc[0]), fVertex>::value);

        auto g_iv_acc = graph.template get_access<sycl::access::mode::read_write, iVertex>(h);
        static_assert(std::is_same<decltype(g_iv_acc[0]), iVertex>::value);

        auto g_i_f_e_acc = graph.template get_access<sycl::access::mode::read_write, typename decltype(i_f_e_buf)::Edge_t>(h);
        static_assert(std::is_same<decltype(g_i_f_e_acc[0]), i_f_edge_t>::value);

        auto g_f_i_e_acc = graph.template get_access<sycl::access::mode::read_write, typename decltype(f_i_e_buf)::Edge_t>(h);
        static_assert(std::is_same<decltype(g_f_i_e_acc[0]), f_i_edge_t>::value);


    });


    // std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges() << " edges." << std::endl;

    // auto& i_f_e_buf_ref = graph.edge_buf.get_edges<decltype(i_f_e_buf)>();
    // static_assert(std::is_same<decltype(i_f_e_buf_ref), decltype(i_f_e_buf)>::value);

    return 0;
}