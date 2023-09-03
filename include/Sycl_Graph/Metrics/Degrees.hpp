#ifndef SYCL_GRAPH_DEGREES_HPP
#define SYCL_GRAPH_DEGREES_HPP
#include <Sycl_Graph/Common.hpp>
#include <Sycl_Graph/Graph/Graph.hpp>
sycl::event compute_in_degrees(sycl::queue &q, Edgebuf_t<1> &edges, sycl::buffer<uint32_t, 1> &degrees);

sycl::event compute_out_degrees(sycl::queue &q, Edgebuf_t<1> &edges, sycl::buffer<uint32_t, 1> &degrees);

sycl::event compute_directed_degrees(sycl::queue &q, Edgebuf_t<1> &edges, sycl::buffer<uint32_t, 1> &in_degrees, sycl::buffer<uint32_t, 1> &out_degrees);

sycl::event compute_undirected_degrees(sycl::queue &q, Edgebuf_t<1> &edges, sycl::buffer<uint32_t, 1> &degrees);

sycl::event compute_undirected_affinity_degrees(sycl::queue& q, Edgebuf_t<1>& edges, sycl::buffer<uint32_t, 1>& vcm, uint32_t N_communities, sycl::buffer<uint32_t, 1>& degrees);

#endif
