#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_HPP
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
namespace Sycl_Graph::Epidemiological
{
    enum SIR_Individual_State_t: char
    {
        SIR_INDIVIDUAL_S = 0,
        SIR_INDIVIDUAL_I = 1,
        SIR_INDIVIDUAL_R = 2
    } ;

    typedef Vertex<SIR_Individual_State_t> SIR_Individual_Vertex_t;
    typedef Edge<float, SIR_Individual_Vertex_t, SIR_Individual_Vertex_t> SIR_Individual_Edge_t;
    typedef Sycl_Graph::Sycl::Edge_Buffer<SIR_Individual_Edge_t> SIR_Individual_Edge_Buffer_t;
    typedef Sycl_Graph::Sycl::Vertex_Buffer<SIR_Individual_Vertex_t> SIR_Individual_Vertex_Buffer_t;
    typedef Sycl_Graph::Sycl::Graph<SIR_Individual_Vertex_Buffer_t, SIR_Individual_Edge_Buffer_t> SIR_Individual_Graph_t;
}
#endif //SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_HPP
