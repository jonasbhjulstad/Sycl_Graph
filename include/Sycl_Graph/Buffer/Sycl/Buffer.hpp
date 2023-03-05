#ifndef SYCL_GRAPH_BUFFER_SYCL_BUFFER_HPP
#define SYCL_GRAPH_BUFFER_SYCL_BUFFER_HPP
#include <Sycl_Graph/Buffer/Base/Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
#include <Sycl_Graph/type_helpers.hpp>
#include <tuple>
#include <algorithm>
namespace Sycl_Graph::Sycl
{

    template <sycl::access::mode Mode, typename... Ds>
    struct Buffer_Accessor
    {
        Buffer_Accessor(sycl::buffer<Ds, 1> &...bufs, sycl::handler &h,
                        sycl::property_list props = {})
            : accessors(std::make_tuple({bufs, h, props} ...))
        {
        }
        template <typename D>
        sycl::accessor<D, 1, Mode> get()
        {
            return std::get<sycl::accessor<D, 1, Mode>>(accessors);
        }
        std::tuple<sycl::accessor<Ds, 1, Mode> ...> accessors;
    };

    template <std::unsigned_integral uI_t, typename... Ds>
    struct Buffer
    {
        typedef std::tuple<Ds...> Data_t;
        static constexpr size_t N_buffers = sizeof...(Ds);
        static constexpr std::array<size_t, N_buffers> buffer_type_sizes = {sizeof(Ds)...};
        sycl::queue &q;
        std::tuple<sycl::buffer<Ds, 1>...> bufs;
        uI_t curr_size = 0;

        Buffer(sycl::queue &q, uI_t N, const sycl::property_list &props = {})
            : bufs(sycl::buffer<Ds, 1>(sycl::range<1>(std::max<uI_t>(N, 1)), props)...), q(q){}

        Buffer(sycl::queue &q, const std::vector<Ds>& ... data,
               const sycl::property_list &props = {})
            : bufs(sycl::buffer<Ds, 1>(data, props)...), q(q), curr_size(std::get<0>(data ...).size()){}

        uI_t current_size() const { return curr_size; }


        // returns a buffer accessor with all types
        template <sycl::access::mode Mode>
        Buffer_Accessor<Mode, Ds...> get_access(sycl::handler &h)
        {
            return Buffer_Accessor<Mode, Ds...>(bufs, h);
        }

        template <sycl::access::mode Mode, typename D>
        sycl::accessor<D, 1, Mode> get_access(sycl::handler &h)
        {
            return std::get<sycl::accessor<D, 1, Mode>>(bufs);
        }

        // returns a buffer accessor with only the specified types
        template <sycl::access::mode Mode, typename... D_subset>
        Buffer_Accessor<Mode, D_subset...> get_access(sycl::handler &h)
        {   
            auto types = indices_of_types<D_subset ..., Ds ...>();
            return Buffer_Accessor<Mode, D_subset...>(get_by_types<D_subset ..., Ds ...>(bufs), h);
        }

        void resize(uI_t new_size)
        {
            buffer_resize(bufs, q, new_size);
            curr_size = std::min(curr_size, new_size);
        }

        template <typename Target_t>
        void assign_add(const std::tuple<std::vector<Ds> ...>& data)
        {
            auto N_added = buffer_assign_add<Target_t, uI_t, Ds ...>(bufs, q, data, curr_size);
            curr_size += N_added;
        }

        template <typename Target_t>
        void assign_add(const std::vector<Ds>& ... data)
        {
            this->assign_add<Target_t>(std::make_tuple(data ...));
        }

        void add(const std::vector<Ds> &... data, uI_t offset = 0)
        {
            this->add(std::make_tuple(data...), offset);
        }

        void assign(const std::vector<Ds> &... data, uI_t offset = 0)
        {
            buffer_assign(bufs, data..., q);
        }

        std::tuple<std::vector<Ds>...> get(uI_t offset, uI_t size)
        {
            return buffer_get(bufs, q, offset, size);
        }

        std::tuple<std::vector<Ds>...> get(const std::vector<uI_t> &indices)
        {
            return buffer_get(bufs, q, indices);
        }

        std::tuple<std::vector<Ds>...> get(auto condition)
        {
            return buffer_get(bufs, q, condition);
        }

        template <typename D>
        void remove_elements(bool (*cond)(const D&))
        {
            auto N_removed = buffer_remove(bufs, q, cond);
            curr_size -= N_removed;
        }

        void remove_elements(uI_t offset, uI_t size)
        {
            auto N_removed = buffer_remove(bufs, q, offset, size);
            curr_size -= N_removed;
        }

        void remove_elements(const std::vector<uI_t> &indices)
        {
            auto N_removed = buffer_remove(bufs, q, indices);
            curr_size -= N_removed;
        }

        Buffer<uI_t, Ds...> &operator=(const Buffer<uI_t, Ds...> &other)
        {
            bufs = other.bufs;
            return *this;
        }

        Buffer<uI_t, Ds...> &operator+(const Buffer<uI_t, Ds...> &other)
        {
            buffer_combine(bufs, other.bufs, q);
            return *this;
        }

        uI_t current_size()
        {
            return curr_size;
        }

        uI_t max_size() const
        {
            return std::get<0>(bufs).size();
        }

        uI_t byte_size() const
        {
            return std::accumulate(buffer_type_sizes.begin(), buffer_type_sizes.end(), 0) * max_size();
        }
    };


} // namespace Sycl_Graph::Sycl
#endif