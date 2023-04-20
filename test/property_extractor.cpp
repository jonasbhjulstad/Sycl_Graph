#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Buffer/Sycl/Base/Vertex_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Invariant/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Invariant/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Invariant/Graph.hpp>

#include <Sycl_Graph/Algorithms/Properties/Sycl/Property_Extractor.hpp>
#include <Sycl_Graph/Algorithms/Properties/Sycl/Degree_Properties.hpp>
#include <iostream>
using namespace Sycl_Graph;
typedef Sycl_Graph::Invariant::Vertex<float> fVertex_t;
typedef Sycl_Graph::Invariant::Vertex<int> iVertex_t;
typedef Sycl_Graph::Invariant::Vertex<double> dVertex_t;
typedef Sycl_Graph::Base::Edge<float> fEdge_t;
typedef Sycl_Graph::Invariant::Edge<fEdge_t, iVertex_t, fVertex_t> i_f_edge_t;
typedef Sycl_Graph::Invariant::Edge<fEdge_t, fVertex_t, iVertex_t> f_i_edge_t;
typedef Sycl_Graph::Invariant::Edge<fEdge_t, iVertex_t, iVertex_t> i_i_edge_t;


int main()
{
    sycl::queue q(sycl::gpu_selector_v);

    std::vector<fVertex_t> fvertices = {fVertex_t{0, 0.0f}, fVertex_t{1, 1.0f}, fVertex_t{2, 2.0f}, fVertex_t{3, 3.0f}};
    std::vector<iVertex_t> ivertices = {iVertex_t{0, 0}, iVertex_t{1, 1}, iVertex_t{2, 2}, iVertex_t{3, 3}};
    std::vector<dVertex_t> dvertices = {dVertex_t{0, 0.0}, dVertex_t{1, 1.0}, dVertex_t{2, 2.0}, dVertex_t{3, 3.0}};


    std::vector<i_f_edge_t> i_f_edges = {i_f_edge_t(0, 1), i_f_edge_t(1, 2), i_f_edge_t(2, 3), i_f_edge_t(3, 2)};
    std::vector<f_i_edge_t> f_i_edges = {f_i_edge_t(0, 1), f_i_edge_t(1, 2), f_i_edge_t(2, 3), f_i_edge_t(3, 2)};
    std::vector<i_i_edge_t> i_i_edges = {i_i_edge_t(0, 1), i_i_edge_t(1, 2), i_i_edge_t(2, 3), i_i_edge_t(3, 2)};

    Sycl_Graph::Sycl::Base::Vertex_Buffer fv_buf(q, fvertices);
    Sycl_Graph::Sycl::Base::Vertex_Buffer iv_buf(q, ivertices);
    Sycl_Graph::Sycl::Base::Edge_Buffer i_f_e_buf(q, i_f_edges);
    Sycl_Graph::Sycl::Base::Edge_Buffer f_i_e_buf(q, f_i_edges);
    Sycl_Graph::Sycl::Base::Edge_Buffer i_i_e_buf(q, i_i_edges);



    Sycl_Graph::Sycl::Invariant::Vertex_Buffer vertex_buffer(fv_buf, iv_buf);
    Sycl_Graph::Sycl::Invariant::Edge_Buffer edge_buffer(i_f_e_buf, f_i_e_buf, i_i_e_buf);
    Sycl_Graph::Sycl::Invariant::Graph graph(vertex_buffer, edge_buffer, q);





    std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges() << " edges." << std::endl;


    Sycl_Graph::Sycl::Degree_Extractor i_f_extractor(iv_buf, fv_buf, i_f_e_buf);
    // Sycl_Graph::Sycl::Degree_Extractor f_i_extractor(fv_buf, iv_buf, f_i_e_buf);


    // auto extractors = std::make_tuple(i_f_extractor_in, i_f_extractor_out);//, f_i_extractor_in, f_i_extractor_out);
    auto extractors = std::make_tuple(i_f_extractor);//, f_i_extractor_in, f_i_extractor_out);

    auto properties = Sycl_Graph::Sycl::extract_properties(graph, extractors, q);

    std::cout << "Degrees:" << std::endl;
    auto printvecpair = [&](const auto& vec) {
        for (const auto& v : vec) {
            std::cout << "From " << v.from.id << ": " << v.from.degree << ", To " << v.to.id << ": " << v.to.degree << std::endl;
        }
        std::cout << std::endl;
    };

    std::apply([&](auto&&... args) {
        (printvecpair(args), ...);
    }, properties);

    

    return 0;
}