#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_ADJACENCY_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_ADJACENCY_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Sycl/Invariant_Graph.hpp>
#include <Sycl_Graph/Algorithms/Properties/Degree_Properties.hpp>
#include <Eigen/Sparse>

namespace Sycl_Graph::Sycl
{
    template <Invariant_Graph_type Graph>
    Eigen::SparseMatrix<uint32_t> get_adjacency_matrix(Graph& G)
    {
        Eigen::SparseMatrix<uint32_t> adj


    Sycl_Graph::Sycl::Directed_Degree_Extractor i_f_extractor(iv_buf, fv_buf, i_f_e_buf);
    Sycl_Graph::Sycl::Directed_Degree_Extractor f_i_extractor(fv_buf, iv_buf, f_i_e_buf);
    Sycl_Graph::Sycl::Directed_Degree_Extractor i_i_extractor(iv_buf, iv_buf, i_i_e_buf);
    std::tuple<std::string, std::string, std::string> edge_type_names= std::make_tuple("integer-float", "float-integer", "integer-integer");

    // auto extractors = std::make_tuple(i_f_extractor_in, i_f_extractor_out);//, f_i_extractor_in, f_i_extractor_out);
    auto extractors = std::make_tuple(i_f_extractor, f_i_extractor, i_i_extractor);

    auto properties = Sycl_Graph::extract_properties(graph, extractors, q);
    auto printvecpair = [&](const auto& vec, const std::string name) {
        std::cout << "Extractor for " << name << " edges" << std::endl;
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

    }
}

#endif