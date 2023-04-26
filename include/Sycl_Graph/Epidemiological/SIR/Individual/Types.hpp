#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_TYPES_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_TYPES_HPP
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>

namespace Sycl_Graph::Epidemiological
{
    enum class: unsigned char
    {
        SIR_INDIVIDUAL_S,
        SIR_INDIVIDUAL_I,
        SIR_INDIVIDUAL_R
    } SIR_Individual_State_t;

    typedef Vertex<SIR_Individual_State_t> SIR_Individual_Vertex_t;
    typedef Edge<float, SIR_Individual_Vertex_t, SIR_Individual_Vertex_t> SIR_Individual_Infection_Edge_t;
    typedef Edge<float, SIR_Individual_Vertex_t, SIR_Individual_Vertex_t> SIR_Individual_Recovery_Edge_t;

}