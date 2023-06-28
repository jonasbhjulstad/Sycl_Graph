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
namespace Sycl_Graph::Sycl
{


    //Buffer accessor contains one sentinel/dummy element
    template <sycl::access::mode Mode, typename... Ds>
    struct Buffer_Accessor
    {

        Buffer_Accessor(sycl::buffer<Ds, 1> &...bufs, sycl::handler &h,
                        sycl::property_list props = {})
            : accessors(sycl::accessor<Ds, 1, Mode>(bufs, h, props)...)
        {
        }

        static Buffer_Accessor<Mode, Ds...> make_accessor(std::tuple<sycl::buffer<Ds, 1> ...> &bufs, sycl::handler &h,
                                                   sycl::property_list props = {})
        {
            return std::apply([&h, &props](auto &...bufs) { return Buffer_Accessor<Mode, Ds...>(bufs..., h, props); }, bufs);
        }

        size_t size() const
        {
            // - 1 accounts for dummy element
            return std::get<0>(accessors).size() - 1;
        }

        std::tuple<Ds...> get_idx(sycl::id<1> i) const
        {
            return std::apply([&i](auto &...accessors) { return std::make_tuple(accessors[i]...); }, accessors);
        }


        std::tuple<Ds...> operator[](sycl::id<1> i) const
        {
            return get_idx(i);
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


    template <typename... Ds>
    struct Buffer
    {
        typedef std::tuple<Ds...> Data_t;
        static constexpr size_t N_buffers = sizeof...(Ds);
        static constexpr std::array<size_t, N_buffers> buffer_type_sizes = {sizeof(Ds)...};
        sycl::queue &q;
        std::tuple<sycl::buffer<Ds, 1>...> bufs;
        uint32_t curr_size = 0;

        Buffer(sycl::queue &q, uint32_t N, const sycl::property_list &props = {})
            : bufs(sycl::buffer<Ds, 1>(sycl::range<1>(std::max<uint32_t>(N, 1)), props)...), q(q){}

        Buffer(sycl::queue &q, const std::vector<Ds>& ... data,
               const sycl::property_list &props = {})
            : bufs(sycl::buffer<Ds, 1>(data, props)...), q(q), curr_size(std::get<0>(bufs).size()){}

        uint32_t current_size() const { return curr_size; }

        template <typename T>
        static constexpr bool is_Data_type = (std::is_same_v<T, Ds> || ...);

        template <typename ... Ts>
        static constexpr bool is_Data_types = (is_Data_type<Ts> && ...);

        template <typename T>
        static constexpr void Data_type_assert()
        {
            static_assert(is_Data_type<T>, "Invalid data type");
        }

        template <typename ... Ts>
        static constexpr void Data_types_assert()
        {
            (Data_type_assert<Ts>(), ...);
        }

        // returns a buffer accessor with only the specified types
        template <sycl::access::mode Mode, typename... D_subset> requires is_Data_types<D_subset...>
        auto get_access(sycl::handler &h)
        {
            return Buffer_Accessor<Mode, D_subset...>(std::get<sycl::buffer<D_subset,1>>(bufs) ..., h);
        }

        auto get_buffers() const
        {
            return bufs;
        }

        template <typename D>
        auto& get_buffer()
        {
            static_assert((std::is_same_v<D, Ds> || ...), "Buffer does not contain type D");
            std::cout << "getting buffer of type " << typeid(D).name() << std::endl;
            return std::get<sycl::buffer<D, 1>>(bufs);
        }

        void resize(uint32_t new_size)
        {
            buffer_resize(bufs, q, new_size);
            curr_size = std::min(curr_size, new_size);
        }


        void add(const std::tuple<std::vector<Ds>...>& data)
        {
            buffer_add(bufs, data, q, curr_size);
            curr_size += std::get<0>(data).size();
        }

        void assign(const std::vector<Ds> &... data)
        {
            buffer_assign(bufs, data..., q);
        }

        std::tuple<std::vector<Ds>...> get(uint32_t offset, uint32_t size)
        {
            return buffer_get(bufs, q, offset, size);
        }

        std::tuple<std::vector<Ds>...> get(const std::vector<uint32_t> &indices)
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

        void remove_elements(uint32_t offset, uint32_t size)
        {
            auto N_removed = buffer_remove(bufs, q, offset, size);
            curr_size -= N_removed;
        }

        void remove_elements(const std::vector<uint32_t> &indices)
        {
            auto N_removed = buffer_remove(bufs, q, indices);
            curr_size -= N_removed;
        }

        Buffer<Ds...> &operator=(const Buffer<Ds...> &other)
        {
            bufs = other.bufs;
            return *this;
        }

        Buffer<Ds...> &operator+(const Buffer<Ds...> &other)
        {
            buffer_combine(bufs, other.bufs, q);
            return *this;
        }

        uint32_t current_size()
        {
            return curr_size;
        }

        uint32_t max_size() const
        {
            return std::get<0>(bufs).size();
        }

        uint32_t byte_size() const
        {
            return std::accumulate(buffer_type_sizes.begin(), buffer_type_sizes.end(), 0) * max_size();
        }
    };
    template <typename T>
    concept Buffer_type = true;

} // namespace Sycl_Graph::Sycl
#endif
