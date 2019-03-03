#pragma once

#include "AudioClient.hpp"
#include "OfflineClient.hpp"
#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"
#include "ParameterSet.hpp"
#include "Result.hpp"
#include "TupleUtilities.hpp"
#include <data/FluidMeta.hpp>
#include <tuple>

namespace fluid
{
namespace client
{

//namespace impl
//{

//template <typename Tuple> class FluidBaseClientImpl
//{
//  static_assert(!isSpecialization<Tuple, std::tuple>(),
//                "Fluid Params: Did you forget to make your params constexpr?");
//};


//template <template <typename...> class Tuple, typename... Ts>
template<typename Params>
class FluidBaseClient //<const Tuple<Ts...>>
{
public:
  
  FluidBaseClient(Params& p):mParams(p){}
  
  template<size_t N> auto& get() const
  {
    return mParams.template get<N>();
  }
  
  template <size_t N, typename T> void set(T&& x, Result *reportage) noexcept
  {
    mParams.template set(std::forward<T>(x),reportage);
  }
  
  size_t audioChannelsIn() const noexcept { return mAudioChannelsIn; }
  size_t audioChannelsOut() const noexcept { return mAudioChannelsOut; }
  size_t controlChannelsIn() const noexcept { return mControlChannelsIn; }
  size_t controlChannelsOut() const noexcept { return mControlChannelsOut; }

  size_t maxControlChannelsOut() const noexcept { return mMaxControlChannelsOut; }
  bool   controlTrigger() const noexcept {return mControlTrigger;}
  
  size_t audioBuffersIn() const noexcept { return mBuffersIn; }
  size_t audioBuffersOut() const noexcept { return mBuffersOut; }

protected:

  void audioChannelsIn(const size_t x) noexcept { mAudioChannelsIn = x; }
  void audioChannelsOut(const size_t x) noexcept { mAudioChannelsOut = x; }
  void controlChannelsIn(const size_t x) noexcept { mControlChannelsIn = x; }
  void controlChannelsOut(const size_t x) noexcept { mControlChannelsOut = x; }
  void maxControlChannelsOut(const size_t x) noexcept {  mMaxControlChannelsOut = x; }
  
  void controlTrigger(const bool x ) noexcept {mControlTrigger = x;}

  void audioBuffersIn(const size_t x) noexcept { mBuffersIn = x; }
  void audioBuffersOut(const size_t x) noexcept { mBuffersOut = x; }

  Params&   mParams;

private:
  size_t     mAudioChannelsIn    = 0;
  size_t     mAudioChannelsOut   = 0;
  size_t     mControlChannelsIn  = 0;
  size_t     mControlChannelsOut = 0;
  size_t     mMaxControlChannelsOut = 0;
  bool       mControlTrigger{false};
  size_t     mBuffersIn          = 0;
  size_t     mBuffersOut         = 0;
};



//template<template<typename,typename...> class C, typename P>
//struct ClientDefinition
//{
//  
//  template<typename...Ts>
//  using Client = C<Ts...>;
//  using Params = P;
//};
//
//template<template<typename,typename...> class Client, typename Params>
//constexpr auto defineClient(const Params& p)
//{
//  return ClientDefinition<Client,decltype(impl::parameterSetFromDescriptors(p))>{};
//}



//} // namespace impl
//
//template <class ParamTuple>
//using FluidBaseClient = typename impl::FluidBaseClientImpl<ParamTuple>;


//template<template <typename...> class Client, typename Params, typename T, typename U =T>
//struct MakeClient
//{
//  using type = Client<impl::ParameterSet<Params>,T,U>;
//};

// Used by hosts for detecting client capabilities at compile time
template <class T> using isNonRealTime = typename std::is_base_of<Offline, T>::type;
template <class T> using isRealTime = std::integral_constant<bool, isAudio<T> || isControl<T>>; 

} // namespace client
} // namespace fluid

