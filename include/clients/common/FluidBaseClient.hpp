#pragma once
#include <tuple>
//#include <utility>

namespace fluid {
namespace client {

namespace impl {
template <typename... Ts> struct ParamValueTypes {
  using type = std::tuple<
                std::pair<
                      std::remove_const_t<typename Ts::first_type::type>,
                       std::tuple<typename Ts::second_type>
                >...>;
 private:
  template <size_t...Is>
  static type createImpl(const std::tuple<Ts...> t, std::index_sequence<Is...>)
  {
      return std::make_tuple(
        std::make_pair(
          std::remove_const_t<typename Ts::first_type::type>{},
          std::get<Is>(t).second
      )...);
  }
public:
  static type create(const std::tuple<Ts...> t)
  {
      return ParamValueTypes::createImpl(t, std::make_index_sequence<sizeof...(Ts)>());
  }
};

template <typename... Ts> class FluidBaseClientImpl {
public:
  using ValueTuple = typename impl::ParamValueTypes<Ts...>::type;

  constexpr FluidBaseClientImpl(const std::tuple<Ts...> params) : mParams(impl::ParamValueTypes<Ts...>::create(params)) {}

  template <size_t N> auto setter() {
    return [this](auto &&x) { std::get<N>(mParams).first = x; };
  }

  template <size_t N> auto get() { return std::get<N>(mParams).first; }

private:
  ValueTuple mParams;
};

template <typename Tuple> struct FluidBaseTemplate;

template <typename... Ts> struct FluidBaseTemplate<const std::tuple<Ts...>> {
  using type = FluidBaseClientImpl<Ts...>;
};
} // namespace impl

template <class Tuple>
using FluidBaseClient = typename impl::FluidBaseTemplate<Tuple>::type;

// template <typename... Ts>
// constexpr FluidClientBase<Ts...> makeClient(std::tuple<Ts...> params) {
//  return {params};
//};

template <template <typename, size_t> class F, typename... Ts,
          std::size_t... Is>
void callOnParams(const std::tuple<Ts...> &t, std::index_sequence<Is...>) {
  auto l = {(F<Ts, Is>(std::get<Is>(t)), 0)...};
}

/// Convert tuple of pairs of descriptors and contrstraints to a tuple of descriptors
/// (Wrappers have no need for the constraints, and it gets a bit 5D chess to deal with them)
template<typename...Ts>
struct ParameterDescriptors
{
  using type = std::tuple<typename Ts::first_type...>;
  
  static type get(const std::tuple<Ts...>& tree)
  {
    return getImpl(tree, std::make_index_sequence<sizeof...(Ts)>()); 
  }
  
  private:
  template<size_t...Is>
  static type getImpl(const std::tuple<Ts...>& tree, std::index_sequence<Is...>)
  {
    return std::make_tuple(std::get<Is>(tree).first...);
  }

};



} // namespace client
} // namespace fluid
