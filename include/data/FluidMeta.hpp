///A place to keep metaprogramming gizmos. Quite probably with links to the stack overflow answer I got them from ;-)
#pragma once

#include <iterator>
#include <type_traits>

namespace fluid {

/****
 All() and Some() are can be used with enable_if_t to check that
 variadic template arguments satisfy some condition
 These are names that stroustrup uses, however C++17 introduces
 'conjunction' & 'disjunction' that seem to do the same, but are defined
 slightly differently

 So, one example of use is to make sure that all variadic arguments can
 be converted to a given type. Let's say size_t, and again use our function
 foo that returns int

 enable_if_t<All(std::is_convertible<size_t, Args>::value...),int>
 foo(){...

 We use these for getting different versions of operator() for arguments
 of indices and of slice specifications.

 Both All() and Some() use the common trick with variadic template args of
 recursing through the list. You'll see in both cases a 'base case' declared
 as constexpr, which is the function without any args. Then, the template below
 calls itself, consuming one more arg from the list at a time.
 ****/
// Base case
constexpr bool all() { return true; }
// Recurse
template <typename... Args> constexpr bool all(bool b, Args... args) {
  return b && all(args...);
}
// Base case
constexpr bool some() { return false; }
// Recurse
template <typename... Args> constexpr bool some(bool b, Args... args) {
  return b || some(args...);
}

/****
 Does the iterator of this type fulfill the given itertator category?
 Used by FluidTensorSlice to ensure that we have at least a ForwardIterator
 in its constructor that takes a range
 ****/
template <typename Iterator, typename IteratorTag>
using IsIteratorType =
    std::is_base_of<IteratorTag,
                    typename std::iterator_traits<Iterator>::iterator_category>;


//Detcting constexpr: https://stackoverflow.com/a/50169108
// p() here could be anything <- well, not really: only certain things are recognised as constant expressions
// Relies on the fact that the narrowing conversion in the first template will be an error except for constant expressions
template<int (*p)()> std::true_type isConstexprImpl(decltype(int{(p(), 0U)}));
template<int (*p)()> std::false_type isConstexprImpl(...);
template<int (*p)()> using is_constexpr = decltype(isConstexprImpl<p>(0));

template<class T, template <typename...> class Template>
struct isSpecialization: std::false_type {};

template<template<typename...> class Template, typename...Args>
struct  isSpecialization<Template<Args...>, Template>: std::true_type {}; 

}
