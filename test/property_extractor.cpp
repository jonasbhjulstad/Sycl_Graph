#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>

#include <Sycl_Graph/Algorithms/Properties/Degree_Properties.hpp>
#include<Sycl_Graph/Algorithms/Properties/Operations.hpp>
#include <iostream>
using namespace Sycl_Graph;
typedef Sycl_Graph::Vertex<float> fVertex_t;
typedef Sycl_Graph::Vertex<int> iVertex_t;
typedef Sycl_Graph::Vertex<double> dVertex_t;
typedef Sycl_Graph::Edge<float> fEdge_t;
typedef Sycl_Graph::Edge<fEdge_t, iVertex_t, fVertex_t> i_f_edge_t;
typedef Sycl_Graph::Edge<fEdge_t, fVertex_t, iVertex_t> f_i_edge_t;
typedef Sycl_Graph::Edge<fEdge_t, iVertex_t, iVertex_t> i_i_edge_t;


int main()
{
    sycl::queue q(sycl::gpu_selector_v);

    std::vector<fVertex_t> fvertices = {fVertex_t{0, 0.0f}, fVertex_t{1, 1.0f}, fVertex_t{2, 2.0f}, fVertex_t{3, 3.0f}};
    std::vector<iVertex_t> ivertices = {iVertex_t{0, 0}, iVertex_t{1, 1}, iVertex_t{2, 2}, iVertex_t{3, 3}};


    std::vector<i_f_edge_t> i_f_edges = {i_f_edge_t(1, 2),i_f_edge_t(3,2)}; //i_f_edge_t(1, 2), i_f_edge_t(2, 3), i_f_edge_t(3, 2)};
    std::vector<f_i_edge_t> f_i_edges = {f_i_edge_t(1, 2),f_i_edge_t(2,1)}; //f_i_edge_t(1, 2), f_i_edge_t(2, 3), f_i_edge_t(3, 2)};
    std::vector<i_i_edge_t> i_i_edges = {i_i_edge_t(1, 2),i_i_edge_t(1,2)}; //i_i_edge_t(1, 2), i_i_edge_t(2, 3), i_i_edge_t(3, 2)};

    Sycl_Graph::Sycl::Vertex_Buffer fv_buf(q, fvertices);
    Sycl_Graph::Sycl::Vertex_Buffer iv_buf(q, ivertices);
    Sycl_Graph::Sycl::Edge_Buffer i_f_e_buf(q, i_f_edges);
    Sycl_Graph::Sycl::Edge_Buffer f_i_e_buf(q, f_i_edges);
    Sycl_Graph::Sycl::Edge_Buffer i_i_e_buf(q, i_i_edges);



    Sycl_Graph::Buffer_Pack vertex_buffer(fv_buf, iv_buf);
    Sycl_Graph::Buffer_Pack edge_buffer(i_f_e_buf, f_i_e_buf, i_i_e_buf);
    Sycl_Graph::Sycl::Graph graph(vertex_buffer, edge_buffer, q);





    std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges() << " edges." << std::endl;


    Sycl_Graph::Sycl::Directed_Vertex_Degree_Op i_f_op(iv_buf, fv_buf, i_f_e_buf);
    Sycl_Graph::Sycl::Directed_Vertex_Degree_Op f_i_op(fv_buf, iv_buf, f_i_e_buf);
    Sycl_Graph::Sycl::Directed_Vertex_Degree_Op i_i_op(iv_buf, iv_buf, i_i_e_buf);
    std::tuple<std::string, std::string, std::string> edge_type_names= std::make_tuple("integer-float", "float-integer", "integer-integer");

    // auto ops = std::make_tuple(i_f_op_in, i_f_op_out);//, f_i_op_in, f_i_op_out);
    auto ops = std::make_tuple(i_f_op, f_i_op, i_i_op);

    auto properties = Sycl_Graph::Sycl::apply_operations(graph, ops, q);
    auto printvecpair = [&](const auto& vec, const std::string name) {
        std::cout << "op for " << name << " edges" << std::endl;
        for (const auto& v : vec) {
            std::cout << "From: " << v.from << ", To: " << v.to << std::endl;
        }
        std::cout << std::endl;
    };
    uint32_t n = 0;

    std::cout << "Degrees" << std::endl;
    
    std::apply([&](auto&&... name){
    std::apply([&](auto&&... args) {
        // ((std::cout << "Property " << n++ << std::endl), printvecpair(args), ...);
        (printvecpair(args, name), ...);
    }, properties);}, edge_type_names);



    Sycl_Graph::Sycl::Undirected_Vertex_Degree_Op i_f_op_undirected(iv_buf, fv_buf, i_f_e_buf);
    Sycl_Graph::Sycl::Undirected_Vertex_Degree_Op f_i_op_undirected(fv_buf, iv_buf, f_i_e_buf);
    Sycl_Graph::Sycl::Undirected_Vertex_Degree_Op i_i_op_undirected(iv_buf, iv_buf, i_i_e_buf);

    auto ops_undirected = std::make_tuple(i_f_op_undirected, f_i_op_undirected, i_i_op_undirected);

    auto properties_undirected = Sycl_Graph::Sycl::apply_operations(graph, ops_undirected, q);
    auto printvecpair_undirected = [&](const auto& vec, const std::string name) {
        std::cout << "op for " << name << " edges" << std::endl;
        for (const auto& v : vec) {
            std::cout << v << std::endl;
        }
        std::cout << std::endl;
    };
    
    std::apply([&](auto&&... name){
    std::apply([&](auto&&... args) {
        // ((std::cout << "Property " << n++ << std::endl), printvecpair(args), ...);
        (printvecpair_undirected(args, name), ...);
    }, properties_undirected);}, edge_type_names);


    

    return 0;
}