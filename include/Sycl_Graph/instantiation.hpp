#include <type_traits>
#include <utility>
namespace utilities {
    template <typename Pack1, typename Pack2> struct merge;

    template <template <typename...> class P, typename... Ts, typename... Us>
    struct merge<P<Ts...>, P<Us...>> {
        using type = P<Ts..., Us...>;
    };
}

template <std::size_t R, typename Pack, typename TypesIgnored, typename Output, typename = void> struct nPr_h;

template <template <typename...> class P, typename First, typename... Rest, typename... TypesIgnored, typename... Output>
struct nPr_h<0, P<First, Rest...>, P<TypesIgnored...>, P<Output...>> {
    // Just one single pack (which must be wrapped in P so that the resulting merge
    // will give all such single packs, rather than a merge of all the types).
    using type = P<P<Output...>>;
};

template <std::size_t R, template <typename...> class P, typename TypesIgnored, typename Output>
struct nPr_h<R, P<>, TypesIgnored, Output> {
    // No pack can come out of this (permuting R types from nothing).
    using type = P<>;
};

template <std::size_t R, template <typename...> class P, typename First, typename... Rest, typename... TypesIgnored, typename... Output>
struct nPr_h<R, P<First, Rest...>, P<TypesIgnored...>, P<Output...>, std::enable_if_t<(R > sizeof...(Rest) + sizeof...(TypesIgnored))>> {
    // No pack can come out of this (since there are fewer types in
    // P<TypesIgnored..., Rest...> than R).
    using type = P<>;
};

template <std::size_t R, template <typename...> class P, typename First, typename... Rest, typename... TypesIgnored, typename... Output>
struct nPr_h<R, P<First, Rest...>, P<TypesIgnored...>, P<Output...>, std::enable_if_t<(R <= sizeof...(Rest) + sizeof...(TypesIgnored) && R != 0)>> : utilities::merge<
    // Case 1: 'First' is in the permuted pack (note that Output..., First are the
    // types in just one pack).  Now continue to get R-1 more types from
    // TypesIgnored..., Rest... (which are the remaining available types since
    // 'First' is no longer available for the remaining R-1 types, and the ignored
    // types are now P<> since we are starting a new nPr_h call).
    typename nPr_h<R-1, P<TypesIgnored..., Rest...>, P<>, P<Output..., First>>::type,
    // Case 2: 'First' in not in the permuted pack, so try to get R types from
    // Rest... to append to Output...  First is appended to TypesIgnored... since
    // it is now among those types not used.
    typename nPr_h<R, P<Rest...>, P<TypesIgnored..., First>, P<Output...>>::type
> { };

template <std::size_t R, typename Pack> struct Permutation_Instantiate;

template <std::size_t R, template <typename...> class P, typename... Ts>
struct Permutation_Instantiate<R, P<Ts...>> : nPr_h<R, P<Ts...>, P<>, P<>> { };
