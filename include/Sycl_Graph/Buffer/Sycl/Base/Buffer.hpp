#ifndef SYCL_GRAPH_BUFFER_SYCL_BASE_BUFFER_HPP
#define SYCL_GRAPH_BUFFER_SYCL_BASE_BUFFER_HPP
#include <Sycl_Graph/Buffer/Base/Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
#include <Sycl_Graph/type_helpers.hpp>
#include <concepts>
#include <type_traits>
#include <tuple>
#include <algorithm>
namespace Sycl_Graph::Sycl::Base
{

    template <sycl::access::mode Mode, typename... Ds>
    struct Buffer_Accessor
    {

        Buffer_Accessor(sycl::buffer<Ds, 1> &...bufs, sycl::handler &h,
                        sycl::property_list props = {})
            : accessors(std::make_tuple({bufs, h, props} ...))
        {
        }

        static Buffer_Accessor<Mode, Ds...> make_accessor(std::tuple<sycl::buffer<Ds, 1> ...> &bufs, sycl::handler &h,
                                                   sycl::property_list props = {})
        {
            return std::apply([&h, &props](auto &...bufs) { return Buffer_Accessor<Mode, Ds...>(bufs..., h, props); }, bufs);
        }

        size_t size() const
        {
            return std::get<0>(accessors).size();
        }

        template <typename D>
        sycl::accessor<D, 1, Mode> get()
        {
            return std::get<sycl::accessor<D, 1, Mode>>(accessors);
        }
        std::tuple<sycl::accessor<Ds, 1, Mode> ...> accessors;
    };

    template <sycl::access::mode Mode, typename D>
    struct Buffer_Accessor<Mode, D>: sycl::accessor<D, 1, Mode>
    {
        Buffer_Accessor(sycl::buffer<D, 1> &buf, sycl::handler &h,
                        sycl::property_list props = {})
            : sycl::accessor<D, 1, Mode>(buf, h, props)
        {
        }
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

        // template <typename T>
        // static constexpr bool is_Buffer_Type = std::disjunction_v<std::is_same<T, Bs>...>;

        // returns a buffer accessor with only the specified types
        template <sycl::access::mode Mode, typename... D_subset>
        auto get_access(sycl::handler &h)
        {   
            if constexpr((std::is_same_v<D_subset, void> || ...))
            {
                return Buffer_Accessor<Mode, Ds ...>::make_accessor(bufs, h);
            }
            else
            {
                return Buffer_Accessor<Mode, D_subset...>(get_by_types<D_subset ..., Ds ...>(bufs), h);
            }
        }

        auto get_buffers() const
        {
            return bufs;
        }

        template <typename D>
        auto& get_buffer()
        {
            return std::get<sycl::buffer<D, 1>>(bufs);
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
    template <typename T>
    concept Buffer_type = true;

} // namespace Sycl_Graph::Sycl
#endif