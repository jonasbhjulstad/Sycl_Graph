#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_SYCL_GRAPH_TYPES_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_SYCL_GRAPH_TYPES_HPP
#include <Sycl_Graph/Epidemiological/SIR/Individual/Types.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <Sycl_Graph/Graph/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Edge_Buffer.hpp>
namespace Sycl_Graph::Epidemiological
{
    typedef Vertex_Buffer<SIR_Individual_Vertex_t> SIR_Individual_Vertex_Buffer_t;
    typedef Edge_Buffer<SIR_Individual_Infection_Edge_t> SIR_Individual_Infection_Edge_Buffer_t;
    typedef Edge_Buffer<SIR_Individual_Recovery_Edge_t> SIR_Individual_Recovery_Edge_Buffer_t;
    // typedef Buffer_Pack<SIR_Individual_Vertex_Buffer_t> SIR_Individual_Vertex_Buffer_Pack_t;
    // typedef Buffer_Pack<SIR_Individual_Infection_Edge_Buffer_t> SIR_Individual_Edge_Buffer_Pack_t;
    typedef Graph<SIR_Individual_Vertex_Buffer_t, SIR_Individual_Edge_Buffer_t> SIR_Individual_Graph_t;


}

#endif