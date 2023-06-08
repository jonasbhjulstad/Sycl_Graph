import Sycl.Graph;
import Base.Graph_Types;
import Sycl.Buffer.Vertex;
import Sycl.Buffer.Edge;
enum SIR_Individual_State_t : unsigned char {
  SIR_INDIVIDUAL_S,
  SIR_INDIVIDUAL_I,
  SIR_INDIVIDUAL_R
};

typedef Vertex<SIR_Individual_State_t> SIR_Individual_Vertex_t;
typedef Edge<float, SIR_Individual_Vertex_t, SIR_Individual_Vertex_t>
    SIR_Individual_Infection_Edge_t;
typedef Sycl::Edge_Buffer<SIR_Individual_Infection_Edge_t> SIR_Individual_Edge_Buffer_t;
typedef Sycl::Vertex_Buffer<SIR_Individual_Vertex_t> SIR_Individual_Vertex_Buffer_t;
typedef Sycl::Graph<SIR_Individual_Vertex_Buffer_t, SIR_Individual_Edge_Buffer_t>
    SIR_Individual_Graph_t;
#endif  // SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_HPP