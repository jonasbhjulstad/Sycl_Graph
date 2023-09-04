#ifndef SYCL_GRAPH_GRAPH_HPP
#define SYCL_GRAPH_GRAPH_HPP
#include <Sycl_Graph/Common.hpp>
namespace Sycl_Graph
{
struct Edge_t: public std::pair<uint32_t, uint32_t>
{
    using Base_t = std::pair<uint32_t, uint32_t>;
    static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
    Edge_t(uint32_t from, uint32_t to): std::pair<uint32_t, uint32_t>(from, to) {}
    Edge_t(): std::pair<uint32_t, uint32_t>(invalid_id, invalid_id) {}
    bool is_valid() const
    {
        return this->first != invalid_id && this->second != invalid_id;
    }
    bool operator==(const Edge_t& other) const
    {
        return this->first == other.first && this->second == other.second;
    }

    bool operator<(const Edge_t& other) const
    {
        return this->first < other.first || (this->first == other.first && this->second < other.second);
    }
    uint32_t from() const
    {
        return this->first;
    }
    uint32_t to() const
    {
        return this->second;
    }

};

template <std::size_t N = 1>
using Edgebuf_t = sycl::buffer<Edge_t, N>;
} // namespace Sycl_Graph
#endif
