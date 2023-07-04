#include <Sycl_Graph/Epidemiological/Epidemiological.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>

template struct SIR_Individual_Recovery<SIR_Individual_Vertex_t, SIR_Individual_State_t>;
template struct SIR_Individual_Recovery<SIR_Individual_State_t, SIR_Individual_State_t>;
template struct SIR_Individual_Recovery<SIR_Individual_Vertex_t, SIR_Individual_Vertex_t>;
template struct SIR_Individual_Recovery<SIR_Individual_State_t, SIR_Individual_State_t>;

template struct SIR_Individual_Infection<SIR_Individual_Vertex_t, SIR_Individual_State_t,
                                         SIR_Individual_Edge_t>;
template struct SIR_Individual_Infection<SIR_Individual_State_t, SIR_Individual_State_t,
                                         SIR_Individual_Edge_t>;
template struct SIR_Individual_Infection<SIR_Individual_Vertex_t, SIR_Individual_Vertex_t,
                                         SIR_Individual_Edge_t>;
template struct SIR_Individual_Infection<SIR_Individual_State_t, SIR_Individual_Vertex_t,
                                         SIR_Individual_Edge_t>;

template struct SIR_Individual_Population_Count<SIR_Individual_Vertex_t>;
template struct SIR_Individual_Population_Count<SIR_Individual_State_t>;
