    #ifndef SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_HPP
    #define SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_HPP
    #include <Sycl_Graph/Buffer/Base/Buffer.hpp>
    #include <Sycl_Graph/Graph/Invariant/Graph_Types.hpp>
    #include <tuple>
    namespace Sycl_Graph::Invariant
    {
    template <Sycl_Graph::Base::Buffer_type ... Bs>
    struct Buffer
    {
        typedef typename std::tuple_element_t<0, std::tuple<Bs ...>>::uI_t uI_t;
        typedef std::tuple<typename Bs::Data_t ...> Data_t;
        static constexpr uI_t N_buffers = sizeof...(Bs);
        Buffer() = default;
        Buffer(const Bs &... buffers): buffers(std::make_tuple(buffers ...)) {}
        Buffer(const Bs &&... buffers): buffers(std::make_tuple(buffers ...)) {}
        Buffer(const std::tuple<Bs ...>& buffers): buffers(buffers) {}
        Buffer(const Buffer &other): buffers(other.buffers) {}

        typedef Buffer<Bs ...> This_t;
        std::tuple<Bs ...> buffers;

        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
        
        template <typename T>
        struct Data_type
        {
            static constexpr bool value = (std::is_same_v<T, typename Bs::Data_t> || ...);
        };
        template <typename D> requires Data_type<D>::value
        static constexpr auto get_buffer_index()
        {
            return Sycl_Graph::index_of_type<D, typename Bs::Data_t ...>();
        }

        template <typename ... Ds> requires (Data_type<Ds>::value && ...)
        static constexpr auto get_buffer_index()
        {
            return std::array<uI_t, sizeof...(Ds)>{type_index<Ds>() ...};
        }

        template <typename D> requires Data_type<D>::value
        auto&& get_buffer()
        {
            return std::get<D>(buffers);
        }
        template <typename ... Ds> requires (Data_type<Ds>::value && ...)
        auto&& get_buffers()
        {
            return std::array{get_buffer<Ds>() ...};
        }

        auto size() const
        {
            return std::apply([](auto &&... buffers) {
                return (buffers.size() + ...);
            }, buffers);
        }

        template <typename D> requires Data_type<D>::value
        auto size() const
        {
            return get_buffer<D>().size();
        }

        template <typename ... Ds> requires (Data_type<Ds>::value && ...)
        void add(const std::vector<Ds> && ... data)
        {
            (get_buffer<Ds>().add(data), ...);
        }

        template <typename ... Ds> requires (Data_type<Ds>::value && ...)
        void remove(const std::vector<Ds>&&... elements)
        {
            ((get_buffer<Ds>().remove(elements), ...));
        }

        auto &operator=(This_t &&other)
        {
            buffers = std::move(other.buffers);
            return *this;
        }

        auto copy() const
        {
            Buffer B;
            B.buffers = this->buffers;
            return B;
        }

        auto &operator+(const This_t &other)
        {
            std::apply([&other](auto &&... buffers) {
                return std::make_tuple((buffers + other.buffers) ...);}, this->buffers);
            return *this;
        }

        template <typename D> requires Data_type<D>::value
        void resize(const uI_t &size)
        {
            get_buffer<D>().resize(size);
        }

        template <typename D> requires Data_type<D>::value
        uI_t current_size() const
        {
            return get_buffer<D>().current_size();
        }

        uI_t current_size() const
        {
            return std::apply([](auto... args) { return (args.current_size() + ...); }, buffers); 
        }

        uI_t max_size() const
        {
            return std::apply([](auto... args) { return (args.max_size() + ...); }, buffers);
        }
    };

    template <typename T>
    concept Buffer_type = Sycl_Graph::Base::Buffer_type<T>;
    } // namespace Sycl_Graph::Invariant

    #endif // SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_HPP