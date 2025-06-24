/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

namespace fluid {
namespace client {

namespace impl {
////////////////////////////////////////////////////
/// Filtering indices from tuples based on predicates

/// When input param true, expose index seq of I
/// otherwise empty index_seq
template <size_t I, bool>
struct Filter;

template <size_t I>
struct Filter<I, true>
{
  using type = std::index_sequence<I>;
};

template <size_t I>
struct Filter<I, false>
{
  using type = std::index_sequence<>;
};


/// Joining index_seqs together.
/// This is heavily based on how Eric Neibler's meta does it, except I didn't
/// have the patience to go all the way up to a ten element list (the longer
/// ones help compile times) Long seqs happen by inheritence, is the punchline.
/// Maybe this generates fewer intermediate types than recursion would. ?
template <typename... Ts>
struct Join
{};

template <>
struct Join<>
{
  using type = std::index_sequence<>;
};

template <size_t... Is>
struct Join<std::index_sequence<Is...>>
{
  using type = std::index_sequence<Is...>;
};

template <size_t... Is, size_t... Js>
struct Join<std::index_sequence<Is...>, std::index_sequence<Js...>>
{
  using type = std::index_sequence<Is..., Js...>;
};

template <size_t... Is, size_t... Js, typename... Rest>
struct Join<std::index_sequence<Is...>, std::index_sequence<Js...>, Rest...>
    : Join<std::index_sequence<Is..., Js...>, Rest...>
{};

/// And from these humble ingredients, we can filter a tuple's indices
/// The special sauce needed is a struct for the Op parameter that exposes a
/// template alias called apply, which will do the std::true_type /
/// std::false_type thing (or anything that exposes a bool called ::value, I'm
/// not proud). E.g struct IsInt { template<typename T> using apply =
/// std::is_same<int,T>;
///};
/// Then use this as an alias on your tuple type

template <typename... Args>
struct FilterTupleIndices;

template <typename Op, template <typename...> class List, typename... Args,
          size_t... Is>
struct FilterTupleIndices<Op, List<Args...>, std::index_sequence<Is...>>
{
  template <typename T>
  using call = typename Op::template apply<T>;

  using type = typename Join<
      typename Filter<Is, call<std::decay_t<Args>>::value>::type...>::type;
};

template <typename, typename>
struct JoinOffsetSequence;

template <size_t... Is, size_t... Js>
struct JoinOffsetSequence<std::index_sequence<Is...>,
                          std::index_sequence<Js...>>
{
  using type = std::index_sequence<Is..., (Js + sizeof...(Is))...>;
};

template <size_t, typename>
struct OffsetSequence;

template <size_t O, size_t... Is>
struct OffsetSequence<O, std::index_sequence<Is...>>
{
  using type = std::index_sequence<(Is + O)...>;
};

template <size_t, typename>
struct SplitIndexSequence;

template <size_t N, size_t... Is>
struct SplitIndexSequence<N, std::index_sequence<Is...>>
{
  using type = typename OffsetSequence<
      N, std::make_index_sequence<(sizeof...(Is) - N)>>::type;
};

template <size_t N, typename Tuple, size_t... Is>
constexpr auto RefTupleFromImpl(Tuple& t, std::index_sequence<Is...>)
{
  return std::tie(std::get<Is>(t)...);
}

template <size_t N, typename Tuple>
constexpr auto RefTupleFrom(Tuple& t)
{
  return RefTupleFromImpl<N>(
      t,
      typename SplitIndexSequence<
          N, std::make_index_sequence<std::tuple_size<Tuple>::value>>::type());
}

template <typename T>
constexpr size_t zeroAll()
{
  return 0u;
}

template <typename... Ts>
using zeroSequenceFor = std::index_sequence<zeroAll<Ts>()...>;


template<typename T>
constexpr std::tuple<T> Include(T& t, std::true_type)
{
  return std::make_tuple(t);
}

template<typename T>
constexpr std::tuple<> Include(T&, std::false_type)
{
  return std::make_tuple();
}

template<size_t N,typename...Ts,size_t...Is>
constexpr auto tupleHead_impl(const std::tuple<Ts...>& t,std::index_sequence<Is...>)
{
  return std::tuple_cat(Include(std::get<Is>(t),std::integral_constant<bool, (Is < N)>{})...);
}

template<size_t N,typename...Ts>
constexpr auto tupleHead(const std::tuple<Ts...>& t)
{
  return tupleHead_impl<N>(t,std::make_index_sequence<sizeof...(Ts)>{});
}

template<size_t N,typename...Ts,size_t...Is>
constexpr auto tupleTail_impl(const std::tuple<Ts...>& t,std::index_sequence<Is...>)
{
  return std::tuple_cat(Include(std::get<Is>(t),std::integral_constant<bool,(Is >= N)>{})...);
}

template<size_t N,typename...Ts>
constexpr auto tupleTail(const std::tuple<Ts...>& t)
{
  return tupleTail_impl<N>(t,std::make_index_sequence<sizeof...(Ts)>{});
}

template<size_t N,typename T, typename...Ts>
constexpr auto tupleInsertAfter(const std::tuple<Ts...>& t, T&& x)
{
  return std::tuple_cat(tupleHead<N>(t), std::make_tuple(std::forward<T>(x)),
                        tupleTail<N>(t));
}

} // namespace impl
} // namespace client
} // namespace fluid
