export module Epidemiological;
//SIR_Individual
#include <Sycl_Graph/Epidemiological/SIR_Individual/Types.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Individual/Operations.hpp>

template class Vertex<SIR_Individual_State_t>;
template class Edge<float, SIR_Individual_Vertex_t, SIR_Individual_Vertex_t>;
template class Edge_Buffer<SIR_Individual_Infection_Edge_t> SIR_Individual_Edge_Buffer_t;
template class Vertex_Buffer<SIR_Individual_Vertex_t> SIR_Individual_Vertex_Buffer_t;
template class Graph<SIR_Individual_Vertex_Buffer_t, SIR_Individual_Edge_Buffer_t>;
template class SIR_Individual_Recovery_Op<SIR_Individual_Vertex_Buffer_t>;
template class SIR_Individual_Infection_Op<SIR_Individual_Edge_Buffer_t, SIR_Individual_Vertex_Buffer_t>;
