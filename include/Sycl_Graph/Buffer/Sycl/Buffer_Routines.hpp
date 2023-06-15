#ifndef SYCL_GRAPH_Buffer_Routines_HPP
#define SYCL_GRAPH_Buffer_Routines_HPP
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>
namespace Sycl_Graph {
  // using namespace oneapi::dpl::experimental;

  template <typename T>
  void buffer_print(sycl::buffer<T, 1> &buf, sycl::queue &q, const std::string &name = "") {
    if (buf.size() == 0) {
      return;
    }

    // if T is integral or floating point
    if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
      std::string type_name = typeid(T).name();
      std::cout << ((name == "") ? type_name : name) << ": ";
      q.submit([&](sycl::handler &h) {
         auto acc = buf.template get_access<sycl::access::mode::read>(h);
         sycl::stream out(1024, 256, h);
         h.single_task([=]() {
           for (int i = 0; i < acc.size(); i++) {
             out << acc[i] << ", ";
           }
         });
       }).wait();
      std::cout << std::endl;
    }
    return;
  }

  template <typename... Ts>
  void buffer_print(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q) {
    std::apply([&](auto &...buf) { (buffer_print(buf, q), ...); }, bufs);
  }

  template <typename T>
  void buffer_resize(sycl::buffer<T, 1> &buf, sycl::queue &q, size_t new_size) {
    if (buf.size() == new_size) {
      return;
    }
    sycl::buffer<T, 1> new_buf(new_size);
    auto smallest_buf_size = std::min<size_t>(buf.size(), new_size);
    q.submit([&](sycl::handler &h) {
      auto acc = buf.template get_access<sycl::access::mode::read>(h);
      auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(smallest_buf_size, [=](sycl::id<1> i) { new_acc[i] = acc[i]; });
    });
    buf = new_buf;
  }

  template <typename... Ts>
  sycl::event buffer_resize(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q, size_t new_size) {
    std::tuple<sycl::buffer<Ts, 1>...> new_bufs = std::make_tuple(sycl::buffer<Ts, 1>(new_size)...);
    auto event = q.submit([&](sycl::handler &h) {
      auto src_accs = std::make_tuple(
          std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read>(h)...);
      auto dest_accs = std::make_tuple(
          std::get<sycl::buffer<Ts, 1>>(new_bufs).template get_access<sycl::access::mode::write>(
              h)...);
      h.parallel_for(new_size, [=](sycl::id<1> i) {
        std::apply(
            [&](auto &...dest_acc) {
              std::apply([&](auto &...src_acc) { ((dest_acc[i] = src_acc[i]), ...); }, src_accs);
            },
            dest_accs);
      });
    });

    bufs = new_bufs;
    return event;
  }

  template <typename T>
  inline void buffer_copy(sycl::buffer<T, 1> &buf, sycl::queue &q, const std::vector<T> &vec) {
    if (vec.size() == 0) {
      return;
    }
    sycl::buffer<T, 1> tmp_buf(vec.data(), sycl::range<1>(vec.size()));
    q.submit([&](sycl::handler &h) {
      auto acc = buf.template get_access<sycl::access::mode::write>(h);
      auto tmp_acc = tmp_buf.template get_access<sycl::access::mode::read>(h);
      h.parallel_for(vec.size(), [=](sycl::id<1> i) { acc[i] = tmp_acc[i]; });
    });
  }

  template <typename T>
  void buffer_add(sycl::buffer<T, 1> &dest_buf, sycl::buffer<T, 1> src_buf, sycl::queue &q,
                  uint32_t offset = 0) {
    if (src_buf.size() == 0) {
      return;
    }
    if constexpr (sizeof(T) > 0) {
      if (dest_buf.size() < src_buf.size() + offset) {
        buffer_resize(dest_buf, q, src_buf.size() + offset);
      }

      q.submit([&](sycl::handler &h) {
        auto src_acc = src_buf.template get_access<sycl::access::mode::read>(h);
        auto dest_acc = dest_buf.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(src_buf.size(), [=](sycl::id<1> i) { dest_acc[i + offset] = src_acc[i]; });
      });
    }
  }

  template <typename... Ts>
  void buffer_add(std::tuple<sycl::buffer<Ts, 1>...> &dest_bufs,
                  std::tuple<sycl::buffer<Ts, 1>...> &src_bufs, sycl::queue &q, uint32_t offset = 0) {
    auto bufsize = std::get<0>(dest_bufs).size();
    static_assert(std::conjunction_v<std::bool_constant<(sizeof(Ts) != 0)>...>,
                  "All buffer types must have non-zero size.");
    if (bufsize == 0) {
      return;
    }
    sycl::event resize_event;
    if (bufsize < offset + std::get<0>(src_bufs).size()) {
      resize_event = buffer_resize(dest_bufs, q, bufsize + std::get<0>(src_bufs).size());
    }
    //get buffer sizes
    auto dest_buf_sizes = std::apply(
        [&](auto &...dest_bufs) { return std::make_tuple(dest_bufs.size()...); }, dest_bufs);
    auto src_buf_sizes = std::apply(
        [&](auto &...src_bufs) { return std::make_tuple(src_bufs.size()...); }, src_bufs);


    q.submit([&](sycl::handler &h) {
      h.depends_on(resize_event);
      auto dest_accs = std::apply(
          [&](auto &...dest_bufs) {
            return std::make_tuple(dest_bufs.template get_access<sycl::access::mode::write>(h)...);
          },
          dest_bufs);
      auto src_accs = std::apply(
          [&](auto &...src_bufs) {
            return std::make_tuple(src_bufs.template get_access<sycl::access::mode::read>(h)...);
          },
          src_bufs);
      h.parallel_for(std::get<0>(src_bufs).size(), [=](sycl::id<1> i) {
        std::apply(
            [&](auto &...dest_acc) {
              std::apply([&](auto &...src_acc) { ((dest_acc[i + offset] = src_acc[i]), ...); },
                         src_accs);
            },
            dest_accs);
      });
    });
  }

  template <typename... Ts>
  void buffer_add(std::tuple<sycl::buffer<Ts, 1>...> &dest_bufs,
                  const std::tuple<std::vector<Ts>...> &src_vecs, sycl::queue &q, uint32_t offset = 0) {
    // create buffers for src_vecs
    auto src_bufs = std::apply(
        [&](auto &...src_vecs) {
          return std::make_tuple(
              sycl::buffer<Ts, 1>(src_vecs.data(), sycl::range<1>(src_vecs.size()))...);
        },
        src_vecs);
    buffer_add(dest_bufs, src_bufs, q, offset);
  }

  template <typename T>
  inline void buffer_add(std::vector<sycl::buffer<T, 1> &> &bufs,
                         const std::vector<const std::vector<T> &> &vecs, sycl::queue &q,
                         const std::vector<uint32_t> &offsets) {
    for (uint32_t i = 0; i < vecs.size(); ++i) {
      buffer_add(bufs[i], vecs[i], q, offsets[i]);
    }
  }

  template <typename T>
  void buffer_add(sycl::buffer<T> &buf, const std::vector<T> &data, sycl::queue &q,
                  uint32_t offset = 0) {
    sycl::buffer<T> tmp_buf(data.data(), sycl::range<1>(data.size()));
    buffer_add(buf, tmp_buf, q, offset);
  }

  template <typename T>
  std::vector<T> buffer_get(sycl::buffer<T> &buf) {
    auto buf_acc = buf.get_host_access();
    std::vector<T> res(buf.size());
    for (int i = 0; i < buf.size(); ++i) {
      res[i] = buf_acc[i];
    }
    return res;
  }

  template <typename T>
  std::vector<T> buffer_get(sycl::buffer<T>& buf, sycl::queue& q)
  {
    auto res = std::vector<T>(buf.size());
    auto res_buf = sycl::buffer<T,1>(res.data(), sycl::range<1>(res.size()));
    q.submit([&](sycl::handler& h) {
      auto buf_acc = buf.template get_access<sycl::access::mode::read>(h);
      auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(buf.size(), [=](sycl::id<1> i) { res_acc[i] = buf_acc[i]; });
    }).wait();

    return res;
  }

  template <typename T>
  std::vector<T> buffer_get(sycl::buffer<T, 1> &buf, sycl::queue &q,
                            const std::vector<uint32_t> &indices) {
    auto condition = [&indices](auto i) {
      return std::find(indices.begin(), indices.end(), i) != indices.end();
    };
    return buffer_get(buf, q, condition);
  }

  template <typename... Ts>
  std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                            sycl::queue &q, uint32_t offset = 0, uint32_t size = 0) {
    return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, offset, size)...);
  }

  template <typename... Ts>
  std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                            sycl::queue &q, const std::vector<uint32_t> &indices) {
    return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, indices)...);
  }

  template <typename... Ts>
  std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                            sycl::queue &q, auto condition) {
    return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, condition)...);
  }

  template <typename T>
  std::optional<std::vector<T>> buffer_get(std::shared_ptr<sycl::buffer<T>> buf) {
    return buf ? std::optional<std::vector<T>>(buffer_get(*buf)) : std::nullopt;
  }

  template <typename... Buf_t> auto buffer_get(std::tuple<std::shared_ptr<Buf_t>...> &bufs) {
    return std::apply([&](auto &&...buf) { return std::make_tuple(buffer_get(buf)...); }, bufs);
  }

  template <typename T>
  std::vector<uint32_t> buffer_get_indices(sycl::buffer<T, 1> &buf, sycl::queue &q,
                                       bool (*condition)(uint32_t)) {
    std::vector<uint32_t> res(buf.size());
    sycl::buffer<uint32_t, 1> res_buf(res.data(), sycl::range<1>(buf.size()));
    q.submit([&](sycl::handler &h) {
      auto acc = buf.template get_access<sycl::access::mode::read>(h);
      auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(buf.size(), [=](sycl::id<1> i) {
        if (condition(i)) {
          res_acc[i] = i;
        }
      });
    });
    std::remove_if(res.begin(), res.end(), [](auto i) { return i == 0; });
    return res;
  }

  template <typename T>
  std::vector<uint32_t> buffer_get_indices(sycl::buffer<T, 1> &buf, sycl::queue &q,
                                       const std::vector<T> &elements) {
    if (buf.size() > 0) {
      std::vector<uint32_t> res(elements.size(), std::numeric_limits<uint32_t>::max());
      sycl::buffer<T, 1> elements_buf(elements.data(), sycl::range<1>(elements.size()));
      sycl::buffer<uint32_t, 1> res_buf(res.data(), sycl::range<1>(elements.size()));
      auto event = q.submit([&](sycl::handler &h) {
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        auto elements_acc = elements_buf.template get_access<sycl::access::mode::read>(h);
        auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(buf.size(), [=](sycl::id<1> i) {
          for (auto j = 0; j < acc.size(); j++) {
            if (acc[j] == elements_acc[i]) {
              res_acc[j] = i;
              return;
            }
          }
        });
      });
      event.wait();
      // copy res_buf to res
      auto res_acc = res_buf.template get_access<sycl::access::mode::read>();
      for (auto i = 0; i < res.size(); i++) {
        res[i] = res_acc[i];
      }
      std::erase_if(res, [](uint32_t i) { return i == std::numeric_limits<uint32_t>::max(); });

      return res;
    }
    return {};
  }

  template <typename... Ts>
  void buffer_assign(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q,
                     const std::vector<uint32_t> &indices, const std::tuple<std::vector<Ts>...> &vecs) {
    const auto buf_size = std::get<0>(bufs).size();
    auto src_bufs = std::apply(
        [&](const auto &...vecs) {
          return std::make_tuple(
              (sycl::buffer<Ts, 1>(vecs.data(), sycl::range<1>(vecs.size())))...);
        },
        vecs);
    sycl::buffer<uint32_t, 1> indices_buf(indices.data(), sycl::range<1>(indices.size()));
    q.submit([&](sycl::handler &h) {
      auto dest_accs = std::make_tuple(
          std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read_write>(
              h)...);
      auto src_accs = std::make_tuple(
          std::get<sycl::buffer<Ts, 1>>(src_bufs).template get_access<sycl::access::mode::read>(
              h)...);
      auto indices_acc = indices_buf.template get_access<sycl::access::mode::read>(h);
      h.parallel_for(buf_size, [=](sycl::id<1> i) {
        if (i == indices_acc[i]) {
          std::apply(
              [&](auto &...dest_acc) {
                std::apply([&](auto &...src_acc) { ((dest_acc[i] = src_acc[i]), ...); }, src_accs);
              },
              dest_accs);
        }
      });
    });
  }

  template <typename Target_t, typename... Ts>
  void buffer_assign(const std::vector<Target_t> &target, std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                     sycl::queue &q, const std::vector<Ts> &...vecs) {
    sycl::buffer<Target_t, 1> target_buf(target.data(), sycl::range<1>(target.size()));
    auto indices = buffer_get_indices(target_buf, q, [&](auto t) { return t == target; });
    buffer_assign(bufs, q, indices, vecs...);
  }

  template <typename... Ts>
  uint32_t buffer_assign_add(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q,
                         const std::vector<uint32_t> &indices,
                         const std::tuple<std::vector<Ts>...> &vecs,
                         uint32_t N_max = std::numeric_limits<uint32_t>::max()) {
    const auto buf_size = std::min<uint32_t>(std::get<0>(bufs).size(), N_max);
    if (buf_size > 0 && indices.size() > 0) {
      buffer_assign(bufs, q, indices, vecs);
      buffer_print(bufs, q);
      std::tuple<std::vector<Ts>...> filtered_vecs;
      ((std::get<std::vector<Ts>>(filtered_vecs).reserve(buf_size - indices.size())), ...);
      for (auto i = 0; i < buf_size - indices.size(); i++) {
        if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
          ((std::get<std::vector<Ts>>(filtered_vecs).push_back(std::get<std::vector<Ts>>(vecs)[i])),
           ...);
        }
      }
      buffer_add(bufs, filtered_vecs, q, buf_size);
      return std::get<0>(filtered_vecs).size();

    } else {
      buffer_add(bufs, vecs, q, buf_size);
      return std::get<0>(vecs).size();
    }
  }

  template <typename T>
  sycl::event buffer_fill(sycl::buffer<T> &buf, const T &value, sycl::queue &q) {
    return q.submit([&](sycl::handler &h) {
      auto acc = buf.template get_access<sycl::access::mode::write>(h);
      h.fill(acc, value);
    });
  }

  template <typename Target_t, typename... Ts>
  uint32_t buffer_assign_add(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q,
                         const std::tuple<std::vector<Ts>...> &vecs,
                         uint32_t N_max = std::numeric_limits<uint32_t>::max()) {
    const auto &target = std::get<std::vector<Target_t>>(vecs);
    std::vector<uint32_t> indices;
    if (std::get<0>(bufs).size() > 0) {
      indices = buffer_get_indices(std::get<sycl::buffer<Target_t, 1>>(bufs), q, target);
    }
    // auto target_buffers = tuple_filter<sycl::buffer<Bs, 1> ...>::template filter<sycl::buffer<Ts,
    // 1> ...>(bufs);
    return buffer_assign_add<uint32_t, Ts...>(bufs, q, indices, vecs, N_max);
  }

  // removes elements at offset to offset+size
  template <typename T>
  void buffer_remove(sycl::buffer<T, 1> &buf, sycl::queue &q, uint32_t offset = 0, uint32_t size = 0) {
    if (size == 0) {
      size = buf.size();
    }
    if constexpr (sizeof(T) > 0) {
      q.submit([&](sycl::handler &h) {
        auto acc = buf.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(size, [=](sycl::id<1> i) { acc[i + offset] = acc[i + offset + size]; });
      });
    }
  }

  template <typename... Ts>
  void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q, uint32_t offset = 0,
                     uint32_t size = 0) {
    (buffer_remove(std::get<sycl::buffer<Ts, 1>>(bufs), q, offset, size), ...);
  }

  template <typename... Ts>
  void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q,
                     const std::vector<uint32_t> &indices,
                     uint32_t N_max = std::numeric_limits<uint32_t>::max()) {
    const auto buf_size = std::min<uint32_t>(std::get<0>(bufs).size(), N_max);
    if (buf_size > 0 && indices.size() > 0) {
      auto indices_sorted = indices;
      std::sort(indices_sorted.begin(), indices_sorted.end());
      auto offset = indices_sorted[0];
      auto size = indices_sorted[indices_sorted.size() - 1] - offset + 1;
      buffer_remove(bufs, q, offset, size);
    }
  }

  template <typename... Ts>
  void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q,
                     const std::vector<uint32_t> &indices) {}

  template <typename T, typename... Ts>
  void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q, auto condition) {
    auto indices = buffer_get_indices(std::get<sycl::buffer<T, 1>>(bufs), q, condition);
    buffer_remove(bufs, q, indices);
  }

  template <typename T>
  sycl::buffer<T, 1> buffer_combine(sycl::queue &q, sycl::buffer<T, 1> buf0,
                                    sycl::buffer<T, 1> buf1, uint32_t size0 = 0, uint32_t size1 = 0) {
    if (size0 == 0) {
      size0 = buf0.size();
    }
    if (size1 == 0) {
      size1 = buf1.size();
    }
    sycl::buffer<T, 1> new_buf(size0 + size1);
    q.submit([&](sycl::handler &h) {
      auto acc0 = buf0.template get_access<sycl::access::mode::read>(h);
      auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(size0, [=](sycl::id<1> i) { new_acc[i] = acc0[i]; });
    });
    q.submit([&](sycl::handler &h) {
      auto acc1 = buf1.template get_access<sycl::access::mode::read>(h);
      auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(size1, [=](sycl::id<1> i) { new_acc[i + size0] = acc1[i]; });
    });
    return new_buf;
  }

  template <typename... Ts>
  sycl::buffer<Ts...> buffer_combine(sycl::queue &q, std::tuple<Ts...> bufs, uint32_t size0 = 0,
                                     uint32_t size1 = 0) {
    return std::apply([&](auto &...buf) { return buffer_combine(q, buf..., size0, size1); }, bufs);
  }

  auto generate_seed_buf(uint32_t seed, uint32_t N_seeds, sycl::queue &q) {
    std::mt19937 gen(seed);
    sycl::buffer<uint32_t> seeds(N_seeds);
    // generate random uint32_t numbers
    std::vector<uint32_t> seed_vec(N_seeds);
    std::generate(seed_vec.begin(), seed_vec.end(), gen);
    buffer_add(seeds, seed_vec, q);
    return seeds;
  }

}  // namespace Sycl_Graph
#endif