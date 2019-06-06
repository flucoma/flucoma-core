#pragma once

#include "AudioClient.hpp"
#include "OfflineClient.hpp"
#include "ParameterConstraints.hpp"
#include "ParameterSet.hpp"
#include "ParameterTypes.hpp"
#include "Result.hpp"
#include "TupleUtilities.hpp"
#include <data/FluidMeta.hpp>
#include <tuple>

namespace fluid {
namespace client {

template<typename ParamType, ParamType& PD>
class FluidBaseClient //<const Tuple<Ts...>>
{
public:
  
  using ParamDescType = ParamType;
  using ParamSetType = ParameterSet<ParamDescType>;
  using ParamSetViewType = ParameterSetView<ParamDescType>;
    
  FluidBaseClient(ParamSetViewType& p) : mParams(std::ref(p)){}
  
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
