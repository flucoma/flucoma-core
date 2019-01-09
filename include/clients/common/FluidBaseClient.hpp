#pragma once

#include "AudioClient.hpp"
#include "OfflineClient.hpp"
#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"
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
/// tuple<<pair<Value,tuple<Constraints>>>>
template <typename... Ts> struct ParamValueTypes {

  template <typename T>
  using ValueType = ParameterValue<std::remove_const_t<typename T::first_type>>;

  template <typename T> using ConstraintsType = typename T::second_type;

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
    return std::make_tuple(std::make_pair(ValueType<Ts>{std::get<Is>(t).first},
                                          std::get<Is>(t).second)...);
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

template <size_t N, typename T, typename Params, typename... Constraints>
T clamp(T thisParam, Params &allParams, std::tuple<Constraints...> &c,
        Result *r)
{
  // for each constraint, pass this param,all params
  return clampImpl<N>(thisParam, allParams, c,
                   std::index_sequence_for<Constraints...>(), r);
}

/// FluidBaseClientImpl
/// Common functionality for clients
template <typename T>
std::ostream &operator<<(std::ostream &o, ParameterValue<T> &t)
{
  return o << t.get();
}

template <typename Tuple> class FluidBaseClientImpl
{
  static_assert(!isSpecialization<Tuple, std::tuple>(),
                "Fluid Params: Did you forget to make your params constexpr?");
};

template <typename... Ts> class FluidBaseClientImpl<const std::tuple<Ts...>>
{
public:
  using ValueTuple =
      typename impl::ParamValueTypes<Ts...>::ValuePlusConstraintsType;
  using UnderlyingTypeTuple = std::tuple<typename Ts::first_type::type...>;
  using ParamType           = const typename std::tuple<Ts...>;
  using ParamIndexList      = typename std::index_sequence_for<Ts...>;

  template <template <size_t N, typename T> class Func>
  static void iterateParameters(ParamType params)
  {
    iterateParametersImpl<Func>(params, ParamIndexList());
  }

  constexpr FluidBaseClientImpl(const std::tuple<Ts...> &params) noexcept
      : mParams(impl::ParamValueTypes<Ts...>::create(params))
  {}

  template <size_t N> auto setter(Result *reportage) noexcept
  {
    return [this, reportage](auto &&x) {
      if (reportage) reportage->reset();
      auto  constraints = std::get<N>(mParams).second;
      auto &param       = std::get<N>(mParams).first;
      using ParamType = typename std::remove_reference_t<decltype(param)>::type;
      auto xPrime =
           clamp<N>(static_cast<ParamType>(x), mParams, constraints, reportage);
      param.set(xPrime);
    };
  }

  template <template <size_t N, typename T> class Func, typename Values>
  std::array<Result, sizeof...(Ts)> checkParameterValues(Values *v)
  {
    return checkParameterValuesImpl<Func, Values>(v, ParamIndexList());
  }

  template <template <size_t N, typename T> class Func, typename Values>
  std::array<Result, sizeof...(Ts)> setParameterValues(Values *v, bool reportage)
  {
    return setParameterValuesImpl<Func>(v, ParamIndexList(), reportage);
  }

  template <std::size_t N> auto get() noexcept
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
  
  template <size_t N>
  using ParamTypeAt = typename std::tuple_element<N, ValueTuple>::type::first_type::ParameterType;

  template <typename T, template <size_t, typename> class Func, typename Values,
            size_t N>
  ValueType<T> makeValue(Values *v)
  {
    return {std::get<N>(mParams).first.descriptor(),
            Func<N, ParamTypeAt<N>>()(v)};
  }

  template <template <size_t N, typename T> class Func, typename Values,
            size_t... Is>
  auto checkParameterValuesImpl(Values *v, std::index_sequence<Is...>)
  {
    ValueTuple candidateValues = std::make_tuple(std::make_pair(
        makeValue<Ts, Func, Values, Is>(v), std::get<Is>(mParams).second)...);
    return validateParametersImpl(candidateValues, ParamIndexList());
  }

  template <size_t... Is>
  auto validateParametersImpl(ValueTuple &values, std::index_sequence<Is...>)
  {
    std::array<Result, sizeof...(Is)> results;
    std::initializer_list<int>{
        (clamp<Is>(std::get<Is>(values).first.get(), values,
               std::get<Is>(values).second, &std::get<Is>(results)),
         0)...};
    return results;
  }

  template <template <size_t N, typename T> class Op, size_t... Is>
  static void iterateParametersImpl(ParamType &params,
                                    std::index_sequence<Is...>)
  {
    std::initializer_list<int>{
        (Op<Is, typename Ts::first_type>()(std::get<Is>(params).first), 0)...};
  }

  template <template <size_t N, typename T> class Func, typename Values, size_t...Is>
  auto setParameterValuesImpl(Values* v, std::index_sequence<Is...>, bool reportage)
  {
    static std::array<Result, sizeof...(Ts)> results;
    
    std::initializer_list<int>{(setter<Is>(reportage ? &results[Is] : nullptr)(Func<Is,ParamTypeAt<Is>>()(v)),0)...};
    
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
template <class T>
using isNonRealTime = typename std::is_base_of<Offline, T>::type;
template <class T> using isRealTime = typename std::is_base_of<Audio, T>::type;

} // namespace client
} // namespace fluid

