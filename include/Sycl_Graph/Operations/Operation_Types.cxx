module;
#include <Sycl_Graph/Common/common.hpp>
export module Operations.Types;
export template <typename T> concept Operation_type = true;

export template <typename T> concept Transform_Operation_type
    = Operation_type<T> && requires(T op) {
                             typename T::Source_t;
                             typename T::Target_t;
                             typename T::target_access_mode;
                           };
export template <typename T> concept has_Source = requires(T op) { typename T::Source_t; };

export template <typename T> bool constexpr has_Source_v = has_Source<T>;

export template <typename T> concept has_Target = requires(T op) {
                                                    typename T::Target_t;
                                                    // check that method target_buffer_size exists
                                                  };

export template <typename T> bool constexpr has_Target_v = has_Target<T>;

export struct Operation_Buffer_Void_t {
  char dummy;
};

export template <typename Derived> struct Transform_Operation {
  typedef std::tuple<> Accessor_Types;
  static constexpr std::tuple<> graph_access_modes;
  static constexpr sycl::access_mode target_access_mode = sycl::access_mode::write;
  void _invoke(const auto&, const auto& source_acc, auto& target_acc, sycl::handler& h) const {
    static_cast<const Derived*>(this)->invoke(source_acc, target_acc, h);
  }

  template <typename Graph_t> size_t target_buffer_size(const Graph_t& G) const {
    return static_cast<const Derived*>(this)->target_buffer_size(G);
  }
};
