/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

/// A place to keep metaprogramming gizmos. Quite probably with links to the
/// stack overflow answer I got them from ;-)
#pragma once

#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
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
template <typename... Args>
constexpr bool all(bool b, Args... args)
{
  return b && all(args...);
}
// Base case
constexpr bool some() { return false; }
// Recurse
template <typename... Args>
constexpr bool some(bool b, Args... args)
{
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


// Detcting constexpr: https://stackoverflow.com/a/50169108
// p() here could be anything <- well, not really: only certain things are
// recognised as constant expressions Relies on the fact that the narrowing
// conversion in the first template will be an error except for constant
// expressions
template <int (*p)()>
std::true_type isConstexprImpl(decltype(int{(p(), 0U)}));
template <int (*p)()>
std::false_type isConstexprImpl(...);
template <int (*p)()>
using is_constexpr = decltype(isConstexprImpl<p>(0));

template <class T, template <typename...> class Template>
struct isSpecialization : std::false_type
{};

template <template <typename...> class Template, typename... Args>
struct isSpecialization<Template<Args...>, Template> : std::true_type
{};

////////////////////////////////////////////////////////////////////////////////
// Thank you https://en.cppreference.com/w/cpp/experimental/is_detected

namespace impl {

template <typename... Ts>
using void_t = void;

template <class Default, class AlwaysVoid, template <class...> class Op,
          class... Args>
struct Detector
{
  using value_t = std::false_type;
  using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct Detector<Default, void_t<Op<Args...>>, Op, Args...>
{
  // Note that std::void_t is a C++17 feature
  using value_t = std::true_type;
  using type = Op<Args...>;
};

} // namespace impl


struct Nonesuch
{
  ~Nonesuch() = delete;
  Nonesuch(Nonesuch const&) = delete;
  void operator=(Nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using isDetected =
    typename impl::Detector<Nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using Detected_t = typename impl::Detector<Nonesuch, void, Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using DetectedOr = impl::Detector<Default, void, Op, Args...>;

// Tuple for each
template <typename Tuple, typename F, typename... Args, size_t... Is>
void ForEachImpl(Tuple&& tuple, F&& f, std::index_sequence<Is...>,
                 Args&&... args)
{
  // Nice trick from Louis Dionne makes this more readble;
  using swallow = int[];
  (void) swallow{1, (f(std::get<Is>(std::forward<Tuple>(tuple)),
                       std::forward<Args>(args)...),
                     void(), int{})...};
}

template <typename Tuple, typename F, typename... Args, size_t... Is>
void ForEachIndexImpl(Tuple&& tuple, F&& f, std::index_sequence<Is...>,
                      Args&&... args)
{
  // Nice trick from Louis Dionne makes this more readble;
  using swallow = int[];
  (void) swallow{
      1, (f(std::get<Is>(std::forward<Tuple>(tuple)),
            std::integral_constant<size_t, Is>{}, std::forward<Args>(args)...),
          void(), int{})...};
}

template <typename Tuple, typename F, typename... Args>
void ForEach(Tuple&& tuple, F&& f, Args&&... args)
{
  constexpr size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  ForEachImpl(std::forward<Tuple>(tuple), std::forward<F>(f),
              std::make_index_sequence<N>{}, std::forward<Args>(args)...);
}

template <typename Tuple, typename F, typename Indices, typename... Args>
void ForThese(Tuple&& tuple, F&& f, Indices idx, Args&&... args)
{
  ForEachIndexImpl(std::forward<Tuple>(tuple), std::forward<F>(f), idx,
                   std::forward<Args>(args)...);
}


namespace impl {

template <std::size_t... Is>
constexpr auto indexSequenceReverse(std::index_sequence<Is...> const&)
    -> decltype(std::index_sequence<sizeof...(Is) - 1U - Is...>{});

template <std::size_t N>
using ReverseIndexSequence =
    decltype(indexSequenceReverse(std::make_index_sequence<N>{}));
} // namespace impl

template <typename Tuple, typename F, typename... Args>
void ReverseForEach(Tuple&& tuple, F&& f, Args&&... args)
{
  constexpr size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  ForEachIndexImpl(std::forward<Tuple>(tuple), std::forward<F>(f),
              impl::ReverseIndexSequence<N>{}, std::forward<Args>(args)...);
}


} // namespace fluid
