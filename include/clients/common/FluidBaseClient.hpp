#pragma once

#include "AudioClient.hpp"
#include "OfflineClient.hpp"
#include "ParameterConstraints.hpp"
#include "ParameterSet.hpp"
#include "ParameterTypes.hpp"
#include "MessageSet.hpp"
#include "Result.hpp"
#include "TupleUtilities.hpp"
#include "../../data/FluidMeta.hpp"
#include <tuple>

namespace fluid {
namespace client {

//template<typename ParamType, ParamType& PD, typename MessageType, MessageType& MD>
//class ClientDescriptor
//{
//  using ParamDescType = ParamType;
//  using ParamSetType = ParameterSet<ParamDescType>;
//  using ParamSetViewType = ParameterSetView<ParamDescType>;
//  using MessageSetType = MessageType;
////  using ClientType = FluidBaseClient<ParamType,PD, MessageType,MD>;
//  constexpr static MessageType& getMessageDescriptors() { return MD; }
//  constexpr static ParamDescType& getParameterDescriptors() { return PD; }
////  constexpr ClientDescriptor(ParamType&& p, MessageType&& m): PD{std::move(p)}, MD{std::move{m}} {}
//  constexpr ClientDescriptor(){}
//};
//
////template<typename ParamType, ParamType& PD, typename MessageType, MessageType& MD>
//template<typename Params, typename Messages>
//auto constexpr
//defineClient(Params&& p, Messages&& m)
//{
//  return ClientDescriptor<std::decay_t<decltype(p)>, std::decay_t<decltype(m)>>{p,m};
//}

template<typename ParamType, ParamType& PD, typename MessageType = decltype(NoMessages), MessageType& MD = NoMessages>
class FluidBaseClient
{
public:
  
  using ParamDescType = ParamType;
  using ParamSetType = ParameterSet<ParamDescType>;
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  
  using MessageSetType = MessageType; 
  
  FluidBaseClient(ParamSetViewType& p) : mParams(std::ref(p)){}
  
  constexpr static MessageType getMessageDescriptors() { return MD; }
  
  template<size_t N,typename...Args>
  decltype(auto) invoke(Args&&...args)
  {
    return MD.template invoke<N>(std::forward<Args>(args)...);
  }

  template<size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  template <size_t N, typename T>
  void set(T &&x, Result *reportage) noexcept
  {
    mParams.template set(std::forward<T>(x), reportage);
  }

  size_t audioChannelsIn() const noexcept { return mAudioChannelsIn; }
  size_t audioChannelsOut() const noexcept { return mAudioChannelsOut; }
  size_t controlChannelsIn() const noexcept { return mControlChannelsIn; }
  size_t controlChannelsOut() const noexcept { return mControlChannelsOut; }

  size_t maxControlChannelsOut() const noexcept { return mMaxControlChannelsOut; }
  bool   controlTrigger() const noexcept { return mControlTrigger; }

  size_t audioBuffersIn() const noexcept { return mBuffersIn; }
  size_t audioBuffersOut() const noexcept { return mBuffersOut; }

  constexpr static ParamDescType& getParameterDescriptors() { return PD; }

  const double sampleRate() const noexcept { return mSampleRate; };
  void  sampleRate(double sr) { mSampleRate = sr; }
  
  void setParams(ParamSetViewType& p) { mParams = p; }

protected:
  void audioChannelsIn(const size_t x) noexcept { mAudioChannelsIn = x; }
  void audioChannelsOut(const size_t x) noexcept { mAudioChannelsOut = x; }
  void controlChannelsIn(const size_t x) noexcept { mControlChannelsIn = x; }
  void controlChannelsOut(const size_t x) noexcept { mControlChannelsOut = x; }
  void maxControlChannelsOut(const size_t x) noexcept { mMaxControlChannelsOut = x; }

  void controlTrigger(const bool x) noexcept { mControlTrigger = x; }

  void audioBuffersIn(const size_t x) noexcept { mBuffersIn = x; }
  void audioBuffersOut(const size_t x) noexcept { mBuffersOut = x; }

  std::reference_wrapper<ParamSetViewType>   mParams;

private:
  size_t mAudioChannelsIn       = 0;
  size_t mAudioChannelsOut      = 0;
  size_t mControlChannelsIn     = 0;
  size_t mControlChannelsOut    = 0;
  size_t mMaxControlChannelsOut = 0;
  bool   mControlTrigger{false};
  size_t mBuffersIn  = 0;
  size_t mBuffersOut = 0;
  double mSampleRate = 0;
};

// Used by hosts for detecting client capabilities at compile time
template <class T>
using isNonRealTime = typename std::is_base_of<Offline, T>::type;
template <class T>
using isRealTime = std::integral_constant<bool, isAudio<T> || isControl<T>>;

} // namespace client
} // namespace fluid
