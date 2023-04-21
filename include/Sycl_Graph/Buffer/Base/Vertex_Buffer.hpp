#ifndef SYCL_GRAPH_VERTEX_BUFFER_BASE_HPP
#define SYCL_GRAPH_VERTEX_BUFFER_BASE_HPP
#include <Sycl_Graph/Buffer/Base/Buffer.hpp>
namespace Sycl_Graph
{
    template <Vertex_type V, typename Derived>
    struct Vertex_Buffer
    {
        typedef V Vertex_t;
        typedef typename V::ID_t ID_t;
        typedef typename V::Data_t Data_t;
        typedef Vertex_t Container_t;
        typedef Data_t Container_Data_t;

        auto size() const
        {
            return static_cast<const Derived *>(this)->size();
        }
        void add(const std::vector<Data_t> && data, const std::vector<ID_t> && ids)
        {
            static_cast<Derived *>(this)->add(data, ids);
        }
        std::vector<Vertex_t> get_vertices()
        {
            return static_cast<Derived *>(this)->get_vertices();
        }

        void remove(const std::vector<ID_t>&& ids)
        {
            static_cast<Derived *>(this)->remove(ids);
        }
    };

    template <typename T>
    concept Vertex_Buffer_type = true;
        // Buffer_type<T> &&
        // Vertex_type<typename T::Vertex_t> &&

}
#endif