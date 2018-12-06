#pragma once

#include "AudioClient.hpp"
#include "ParameterTypes.hpp"
#include <tuple>



namespace fluid {
namespace client {

namespace impl {

/// ParamValueTypes
/// Converts a structure of parmeter declarations and constraints to a
/// structure of parameter values and constraints:
/// tuple<<pair<TypeTag,tuple<Constraints>>>> -> tuple<<pair<Value,tuple<Constraints>>>>
template <typename... Ts> struct ParamValueTypes {
  
  template<typename T>
  using value_type = ParameterValue<std::remove_const_t<typename T::first_type>>;
  
  template<typename T>
  using constraints_type = typename T::second_type;
  
  using type = std::tuple<std::pair<value_type<Ts>,constraints_type<Ts>>...>;
 
 private:
  template <size_t...Is>
  static type createImpl(const std::tuple<Ts...>& t, std::index_sequence<Is...>)
  {
      return std::make_tuple(
        std::make_pair(
          value_type<Ts>{std::get<Is>(t).first},
          std::get<Is>(t).second
      )...);
  }
public:
  static type create(const std::tuple<Ts...> descriptors)
  {
//      puts(__PRETTY_FUNCTION__);
      return ParamValueTypes::createImpl(descriptors, std::index_sequence_for<Ts...>());
  }
};

//Clamp value given constraints
//Which are a tuple of
//callable
//tuple<indices of other params>

template <typename T, typename Params, typename Constraints, size_t...Is>
T clampImpl(T thisParam, Params& allParams, Constraints& c, std::index_sequence<Is...>)
{
  T res = thisParam;
// puts(__PRETTY_FUNCTION__);
  (void)std::initializer_list<int>{(res = std::get<Is>(c).clamp(res,allParams),0)...};
  return res;
}


template <typename T, typename Params, typename...Constraints>
T clamp(T thisParam, Params& allParams, std::tuple<Constraints...>& c)
{
  //for each constraint, pass this param,all params
  return clampImpl(thisParam,allParams,c, std::index_sequence_for<Constraints...>());
}
/// FluidBaseClientImpl
/// Common functionality for clients


template<typename T>
std::ostream& operator << (std::ostream& o, ParameterValue<T>& t)
{
    return o << t.get();
}

template <typename... Ts> class FluidBaseClientImpl {
public:
  using ValueTuple = typename impl::ParamValueTypes<Ts...>::type;

  constexpr FluidBaseClientImpl(const std::tuple<Ts...> params) noexcept : mParams(impl::ParamValueTypes<Ts...>::create(params)) {}

  template <size_t N> auto  setter() noexcept {
    return [this](auto &&x) {
      auto constraints = std::get<N>(mParams).second;
      auto param = std::get<N>(mParams).first;
      auto xPrime = clamp(static_cast<typename decltype(param)::type>(x),mParams, constraints);//, mParams, constraints);
    
      std::get<N>(mParams).first.set(xPrime);
      std::cout << std::get<N>(mParams).first << '\n';
    };
    
  }

  template <std::size_t N> auto get() noexcept { return std::get<N>(mParams).first.get(); }
  template <std::size_t N> bool changed() noexcept {
    return std::get<N>(mParams).first.changed();
  }

  //Todo: Could this be made more graceful with tag types? Not without CRTP, I suspect
  size_t audioChannelsIn() const noexcept {return mAudioChannelsIn;}
  size_t audioChannelsOut() const noexcept {return mAudioChannelsOut;}
  size_t controlChannelsIn() const noexcept {return mControlChannelsIn;}
  size_t controlChannelsOut() const noexcept {return mControlChnnalesOut;}
protected:
  void audioChannelsIn(const size_t x) noexcept { mAudioChannelsIn = x;}
  void audioChannelsOut(const size_t x )  noexcept { mAudioChannelsOut = x;}
  void controlChannelsIn(const size_t x)  noexcept { mControlChannelsIn = x;}
  void controlChannelsOut(const size_t x)  noexcept { mControlChnnalesOut = x ;}

private:
  size_t mAudioChannelsIn   = 0;
  size_t mAudioChannelsOut  = 0;
  size_t mControlChannelsIn = 0;
  size_t mControlChnnalesOut= 0;
  ValueTuple mParams;
};

/// FluidBaseTemplate
/// We need a base class templated on Ts... (pairs of parametre types
/// and constraints), but we have a tuple of these things when we declare them
/// This metafunction lets us convert between tuple<Ts...> and Ts...
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
