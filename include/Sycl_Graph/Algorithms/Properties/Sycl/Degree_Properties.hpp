#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#include <Sycl_Graph/Algorithms/Properties/Sycl/Property_Extractor.hpp>
#include <Sycl_Graph/Buffer/Invariant/Buffer.hpp>
#include <CL/sycl.hpp>
namespace Sycl_Graph::Sycl
{
    enum Degree_Property
    {
        In_Degree,
        Out_Degree
    };
    template <Sycl_Graph::Invariant::Vertex_Buffer_type _Vertex_Buffer_t, Sycl_Graph::Invariant::Edge_Buffer_type _Edge_Buffer_t>
    struct Degree_Extractor
    {

        typedef _Vertex_Buffer_t Vertex_Buffer_t;
        typedef _Edge_Buffer_t Edge_Buffer_t;
        typedef typename Edge_Buffer_t::Edge_t Edge_t;
        typedef typename Vertex_Buffer_t::uI_t uI_t;
        typedef typename Vertex_Buffer_t::ID_t ID_t;
        typedef typename Edge_Buffer_t::Edge_t::From_t From_t;
        typedef typename Edge_Buffer_t::Edge_t::To_t To_t;

        typedef ID_t Property_t;
        typedef sycl::accessor<Property_t, 1, sycl::access::mode::read> Property_Access_t;
        typedef std::pair<ID_t, uI_t> Accumulation_Property_t;
        typedef sycl::accessor<Accumulation_Property_t, 1, sycl::access::mode::write> Accumulate_Access_t; 
        Degree_Property property;


        Degree_Extractor(const Vertex_Buffer_t& vertex_buffer, const Edge_Buffer_t& edge_buffer, Degree_Property property = In_Degree): property(property){}

        Property_t apply(const Edge_t& edge_target, const From_t& from, const To_t& to)
        {
            if constexpr (property == In_Degree)
            {
                return edge_target.to;
            }
            else if constexpr (property == Out_Degree)
            {
                return edge_target.from;
            }
        }

        void accumulate(const Accumulation_Property_t& return_properties, Accumulate_Access_t& accumulated_property)
        {
            for (int i = 0; i < return_properties.size(); i++)
            {
                for(int j = 0; j < accumulated_property.size(); j++)
                {
                    if (return_properties[i] == accumulated_property[j].first)
                    {
                        accumulated_property[j].second++;
                    }
                }
            }
        }
    };

    template <Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_t, Sycl_Graph::Base::Edge_Buffer_type _Edge_Buffer_t>
    struct Degree_Square_Sum_Extractor
    {
        typedef _Vertex_Buffer_t Vertex_Buffer_t;
        typedef _Edge_Buffer_t Edge_Buffer_t;
        typedef typename Edge_Buffer_t::Edge_t Edge_t;
        typedef typename Vertex_Buffer_t::uI_t uI_t;
        typedef typename Vertex_Buffer_t::ID_t ID_t;
        typedef typename Edge_Buffer_t::Edge_t::From_t From_t;
        typedef typename Edge_Buffer_t::Edge_t::To_t To_t;

        typedef ID_t Property_t;
        typedef sycl::accessor<Property_t, 1, sycl::access::mode::read> Property_Access_t;
        typedef std::pair<ID_t, uI_t> Accumulation_Property_t;
        typedef sycl::accessor<Accumulation_Property_t, 1, sycl::access::mode::write> Accumulate_Access_t; 
        Degree_Property property;

        Degree_Square_Sum_Extractor(const Vertex_Buffer_t& vertex_buffer, const Edge_Buffer_t& edge_buffer, Degree_Property property = In_Degree): property(property){}


        Property_t apply(const Edge_t& edge_target, const From_t& from, const To_t& to)
        {
            if constexpr (property == In_Degree)
            {
                return edge_target.to;
            }
            else if constexpr (property == Out_Degree)
            {
                return edge_target.from;
            }
        }

        void accumulate(const Accumulation_Property_t& return_properties, Accumulate_Access_t& accumulated_property)
        {
            for (int i = 0; i < return_properties.size(); i++)
            {
                for(int j = 0; j < accumulated_property.size(); j++)
                {
                    if (return_properties[i] == accumulated_property[j].first)
                    {
                        accumulated_property[j].second++;
                    }
                }
            }

            for(int j = 0; j < accumulated_property.size(); j++)
            {
                accumulated_property[j].second = accumulated_property[j].second * accumulated_property[j].second;
            }
        }
    };
}
#endif