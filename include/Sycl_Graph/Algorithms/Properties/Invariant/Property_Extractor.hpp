#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_INVARIANT_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_INVARIANT_PROPERTY_EXTRACTOR_HPP
#include <Sycl_Graph/Graph/Invariant/Graph.hpp>
namespace Sycl_Graph::Invariant {

template <typename T>
concept Property_Extractor_type =
    // Indexable<typename T::Property_t, typename T::Property_Access_t> &&
    Sycl_Graph::Invariant::Edge_type<typename T::Edge_t> &&
    requires(T extractor, typename T:: Property_Access_t,
             typename T::Accumulate_Access_t acc_prop_acc,
             typename T::Edge_t edge_target, typename T::Edge_t::From_t from,
             typename T::Edge_t::To_t to,
             typename T::Accumulation_Property_t property_list) {
      extractor.apply(edge_target, from, to);
      extractor.accumulate(property_list, acc_prop_acc);
      typename T::Accumulation_Property_t;
    };


} // namespace Sycl_Graph::Base

#endif