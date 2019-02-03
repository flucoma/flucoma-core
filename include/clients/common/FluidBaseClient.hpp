#pragma once

#include "AudioClient.hpp"
#include "OfflineClient.hpp"
#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"
#include "TupleUtilities.hpp"
#include <data/FluidMeta.hpp>
#include <tuple>

namespace fluid
{
namespace client
{

namespace impl
{

/// ParamValueTypes
/// Converts a structure of parmeter declarations and constraints to a
/// structure of parameter values and constraints:
/// tuple<<pair<TypeTag,tuple<Constraints>>>> ->

///tuple<
///    tuple<Type, tuple<constraints>, FixedFlag>

/// tuple<<pair<Value,tuple<Constraints>>>>
template <typename... Ts> struct ParamValueTypes {

  template <typename T>
  using ValueType = ParameterValue<std::decay_t<typename std::tuple_element<0,T>::type>>;

  template <typename T> using ConstraintsType = typename std::tuple_element<1,T>::type;

  using ValuePlusConstraintsType =
      std::tuple<std::pair<ValueType<Ts>, ConstraintsType<Ts>>...>;

  static ValuePlusConstraintsType create(const std::tuple<Ts...> descriptors)
  {
    return ParamValueTypes::createImpl(descriptors,
                                       std::index_sequence_for<Ts...>());
  }

private:
  template <size_t... Is>
  static ValuePlusConstraintsType createImpl(const std::tuple<Ts...> &t,
                                             std::index_sequence<Is...>)
  {
    return std::make_tuple(std::make_pair(ValueType<Ts>{std::get<0>(std::get<Is>(t))},
                                          std::get<1>(std::get<Is>(t)))...);
  }
};

// Clamp value given constraints

template <size_t N, typename T, typename Params, typename Constraints, size_t... Is>
T clampImpl(T &thisParam, Params &allParams, Constraints &c,
            std::index_sequence<Is...>, Result *r)
{
  T res = thisParam;
  (void) std::initializer_list<int>{
      (std::get<Is>(c).template clamp<N>(res, allParams, r), 0)...};
  return res;
}

template <typename T>
struct Clamper
{
  template<size_t N,typename Params, typename... Constraints>
  static T clamp(T thisParam, Params &allParams, std::tuple<Constraints...> &c,
        Result *r)
  {
  // for each constraint, pass this param,all params
    return clampImpl<N>(thisParam, allParams, c,
                   std::index_sequence_for<Constraints...>(), r);
  }
};

template<>
struct Clamper<typename BufferT::type>
{
template<size_t N,typename Params, typename... Constraints>
static typename BufferT::type clamp(typename BufferT::type& thisParam, Params&, std::tuple<Constraints...>, Result* r)
{
  return std::move(thisParam);
}
};


/// FluidBaseClientImpl
/// Common functionality for clients
template <typename T>
std::ostream &operator<<(std::ostream &o, ParameterValue<T> &t)
{
  return o << t.get();
}

template<bool B>
struct IsFixed
{
  template<typename T>
  using apply = std::is_same<Fixed<B>, typename std::tuple_element<2, T>::type>;
};

using IsFixedParamTest = IsFixed<true>;
using IsAdjustbleParamTest = IsFixed<false>;


template <typename Tuple> class FluidBaseClientImpl
{
  static_assert(!isSpecialization<Tuple, std::tuple>(),
                "Fluid Params: Did you forget to make your params constexpr?");
};

///Each parameter descriptor in the base client is a three-element tuple
///Third element is flag indicating whether fixed (instantiation only) or not


template <template <typename...> class Tuple, typename... Ts> class FluidBaseClientImpl<const Tuple<Ts...>>
{
public:
  using ValueTuple =
      typename impl::ParamValueTypes<Ts...>::ValuePlusConstraintsType;
  using ParamType           = const typename std::tuple<Ts...>;
  using ParamIndexList      = typename std::index_sequence_for<Ts...>;
  template <size_t N>
  using ParamDescriptorTypeAt = typename std::tuple_element<N, ValueTuple>::type::first_type::ParameterType;
  template <size_t N> using ParamTypeAt = typename ParamDescriptorTypeAt<N>::type;

  using FixedParams   = typename impl::FilterTupleIndices<IsFixedParamTest,std::decay_t<ParamType>,ParamIndexList>::type;
  static constexpr size_t NumFixedParams = FixedParams::size();
  using MutableParams  = typename impl::FilterTupleIndices<IsAdjustbleParamTest,std::decay_t<ParamType>,ParamIndexList>::type;
  static constexpr size_t NumMutableParams = MutableParams::size();


  template<size_t N>
  static auto ParameterDefaultAt(ParamType& params)
  {
      return std::get<0>(std::get<N>(params)).defaultValue;
  }


  template <template <size_t N, typename T> class Func>
  static void iterateParameterDescriptors(ParamType& params)
  {
    iterateParameterDescriptorsImpl<Func>(params, ParamIndexList());
  }
  
  template <template <size_t N, typename T> class Func>
  static void iterateFixedParameterDescriptors(ParamType& params)
  {
    iterateParameterDescriptorsImpl<Func>(params, FixedParams());
  }
  
  template <template <size_t N, typename T> class Func>
  static void iterateMutableParameterDescriptors(ParamType& params)
  {
    iterateParameterDescriptorsImpl<Func>(params, MutableParams());
  }
  
  constexpr FluidBaseClientImpl(const std::tuple<Ts...> &params) noexcept
      : mParams(impl::ParamValueTypes<Ts...>::create(params))
  {}


  template <size_t N, typename T> void set(T&& x, Result *reportage) noexcept
  {
    if (reportage) reportage->reset();
    auto  constraints = std::get<N>(mParams).second;
    auto &param       = std::get<N>(mParams).first;
    using ParamType = typename std::remove_reference_t<decltype(param)>::type;
    auto xPrime =
         Clamper<ParamType>::template clamp<N>(x, mParams, constraints, reportage);
    param.set(std::move(xPrime));
  }

  template <template <size_t N, typename T> class Func, typename...Args>
  std::array<Result, sizeof...(Ts)> checkParameterValues(Args&&...args)
  {
    return checkParameterValuesImpl<Func>(ParamIndexList(),std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename...Args>
  std::array<Result, sizeof...(Ts)> setParameterValues(bool reportage,Args&&...args)
  {
    return setParameterValuesImpl<Func>(ParamIndexList(), reportage,std::forward<Args>(args)...);
  }
  
  template<template <size_t N, typename T> class Func, typename...Args>
  void forEachParam(Args&&...args)
  {
    forEachParamImpl<Func>(ParamIndexList(),std::forward<Args>(args)...);
  }
  

  template <std::size_t N> auto& get() noexcept
  {
    return std::get<N>(mParams).first.get();
  }
  
  template <std::size_t N> bool changed() noexcept
  {
    return std::get<N>(mParams).first.changed();
  }

  // Todo: Could this be made more graceful with tag types? Not without CRTP, I
  // suspect
  size_t audioChannelsIn() const noexcept { return mAudioChannelsIn; }
  size_t audioChannelsOut() const noexcept { return mAudioChannelsOut; }
  size_t controlChannelsIn() const noexcept { return mControlChannelsIn; }
  size_t controlChannelsOut() const noexcept { return mControlChannelsOut; }

  size_t audioBuffersIn() const noexcept { return mBuffersIn; }
  size_t audioBuffersOut() const noexcept { return mBuffersOut; }

protected:

  void audioChannelsIn(const size_t x) noexcept { mAudioChannelsIn = x; }
  void audioChannelsOut(const size_t x) noexcept { mAudioChannelsOut = x; }
  void controlChannelsIn(const size_t x) noexcept { mControlChannelsIn = x; }
  void controlChannelsOut(const size_t x) noexcept { mControlChannelsOut = x; }

  void audioBuffersIn(const size_t x) noexcept { mBuffersIn = x; }
  void audioBuffersOut(const size_t x) noexcept { mBuffersOut = x; }

private:
  template <typename T>
  using ValueType =
      typename impl::ParamValueTypes<Ts...>::template ValueType<T>;
  
//  template <size_t  Is, typename Tuple>
//  using ParamTypeAt = typename std::tuple_element<Is, Tuple>::type;
  
  template<size_t Is, typename VTuple>
  auto& ParamValueAt(VTuple& values)
  {
    return std::get<Is>(values).first.get();
  }
  
  template<size_t Is, typename VTuple>
  auto& ConstraintAt(VTuple& values)
  {
    return std::get<Is>(values).second;
  }
  
  template <typename T, template <size_t, typename> class Func,
            size_t N,typename...Args>
  ValueType<T> makeValue(Args&&...args)
  {
    return {std::get<N>(mParams).first.descriptor(),
            Func<N, ParamDescriptorTypeAt<N>>()(std::forward<Args>(args)...)};
  }


  template <template <size_t N, typename T> class Func,
            size_t... Is,typename...Args>
  auto checkParameterValuesImpl(std::index_sequence<Is...>, Args&&...args)
  {
    ValueTuple candidateValues = std::make_tuple(std::make_pair(
        makeValue<Ts, Func, Is>(std::forward<Args>(args)...), std::get<Is>(mParams).second)...);
    return validateParametersImpl(ParamIndexList(),candidateValues);
  }

  template <size_t... Is>
  auto validateParametersImpl(std::index_sequence<Is...>,ValueTuple &values)
  {
    std::array<Result, sizeof...(Is)> results;
    std::initializer_list<int>{
        (Clamper<ParamTypeAt<Is>>::template clamp<Is>(ParamValueAt<Is>(values), values,
             ConstraintAt<Is>(values), &std::get<Is>(results)),
         0)...};
    return results;
  }
  
  template <template <size_t N, typename T> class Func, typename...Args, size_t...Is>
  void forEachParamImpl(std::index_sequence<Is...>, Args&&...args)
  {
    std::initializer_list<int>{(Func<Is,typename std::tuple_element<0,Ts>::type>()(std::forward<Args>(args)...),0)...};
  }

  template <template <size_t N, typename T> class Op, size_t... Is>
  static void iterateParameterDescriptorsImpl(ParamType &params,
                                    std::index_sequence<Is...>)
  {
    std::initializer_list<int>{
        (Op<Is,  ParamDescriptorTypeAt<Is>>()(std::get<0>(std::get<Is>(params))), 0)...};
  }

  template <template <size_t N, typename T> class Func, typename...Args, size_t...Is>
  auto setParameterValuesImpl(std::index_sequence<Is...>, bool reportage, Args&&...args)
  {
    static std::array<Result, sizeof...(Ts)> results;
    
    std::initializer_list<int>{(set<Is>(Func<Is,ParamDescriptorTypeAt<Is>>()(std::forward<Args>(args)...),reportage ? &results[Is] : nullptr),0)...};
    
    return results;
  }


  size_t     mAudioChannelsIn    = 0;
  size_t     mAudioChannelsOut   = 0;
  size_t     mControlChannelsIn  = 0;
  size_t     mControlChannelsOut = 0;
  size_t     mBuffersIn          = 0;
  size_t     mBuffersOut         = 0;
  ValueTuple mParams;
};

} // namespace impl

template <class ParamTuple>
using FluidBaseClient = typename impl::FluidBaseClientImpl<ParamTuple>;



// Used by hosts for detecting client capabilities at compile time
template <class T> using isNonRealTime = typename std::is_base_of<Offline, T>::type;
template <class T> using isRealTime = std::integral_constant<bool, isAudio<T> || isControl<T>>; 

} // namespace client
} // namespace fluid

