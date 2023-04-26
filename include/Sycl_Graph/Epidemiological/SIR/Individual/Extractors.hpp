#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_EXTRACTORS_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_EXTRACTORS_HPP

namespace Sycl_Graph::Epidemiological
{

    struct Infection_Event_Extractor
    {
        using Accumulate_t = float;
        using Apply_t = uint32_t;
        static constexpr Extractor_Type_t extractor_type = Extractor_Type_Edge;
        using From_t = SIR_Individual_Vertex_t;
        using To_t = SIR_Individual_Vertex_t;
        using Edge_t = SIR_Individual_Infection_Edge_t;

        Accumulate_t accumulate()
        {
            return 
        }
    };
}

#endif
