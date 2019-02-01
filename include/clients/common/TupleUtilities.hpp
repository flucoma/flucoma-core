#pragma once

namespace fluid {
namespace client {

namespace impl {
////////////////////////////////////////////////////
/// Filtering indices from tuples based on predicates

/// When input param true, expose index seq of I
/// otherwise empty index_seq
template <size_t I, bool> struct Filter;

template <size_t I> struct Filter<I, true>
{
  using type = std::index_sequence<I>;
};

template <size_t I> struct Filter<I, false>
{
  using type = std::index_sequence<>;
};

/// Joining index_seqs together.
/// This is heavily based on how Eric Neibler's meta does it, except I didn't have
/// the patience to go all the way up to a ten element list (the longer ones help compile times)
/// Long seqs happen by inheritence, is the punchline. Maybe this generates fewer intermediate types than
/// recursion would. ?
template <typename... Ts> struct Join
{};

template <> struct Join<>
{
  using type = std::index_sequence<>;
};

template <size_t... Is> struct Join<std::index_sequence<Is...>>
{
  using type = std::index_sequence<Is...>;
};

template <size_t... Is, size_t... Js> struct Join<std::index_sequence<Is...>, std::index_sequence<Js...>>
{
  using type = std::index_sequence<Is..., Js...>;
};

template <size_t... Is, size_t... Js, typename... Rest>
struct Join<std::index_sequence<Is...>, std::index_sequence<Js...>, Rest...>
    : Join<std::index_sequence<Is..., Js...>, Rest...>
{};

/// And from these humble ingredients, we can filter a tuple's indices
/// The special sauce needed is a struct for the Op parameter that exposes a template
/// alias called apply, which will do the std::true_type / std::false_type thing (or anything that
/// exposes a bool called ::value, I'm not proud). E.g
/// struct IsInt {
/// template<typename T> using apply = std::is_same<int,T>;
///};
/// Then use this as an alias on your tuple type

template <typename... Args> struct FilterTupleIndices;

template <typename Op, template <typename...> class List, typename... Args, size_t... Is>
struct FilterTupleIndices<Op, List<Args...>, std::index_sequence<Is...>>
{
  template <typename T> using call = typename Op::template apply<T>;
  
  using type = typename Join<typename Filter<Is, call<std::decay_t<Args>>::value>::type...>::type;
};

//template<typename Op,typename Tuple,typename...Args,size_t...Is>
//void forEachInTuple ( Tuple<Args...> a, std::index_sequence<Is...>)
//{
//  (void) std::initializer_list<int>{(Op()(std::forward<Args>(std::get<Is>(a))),0)...};
//}

//template<template <typename,size_t>...RetType, typename Op,typename Tuple,typename...Args,size_t...Is>
//RetType forEachInTuple(Tuple<Args...>& a, std::index_sequence<Is...>)
//{
//  (void) std::initializer_list<int>{(Op()(std::forward<Args>(std::get<Is>(a))),0)...};
//}



} // namespace impl
} // namespace client
} // namespace fluid

