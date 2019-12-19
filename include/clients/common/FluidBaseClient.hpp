/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "AudioClient.hpp"
#include "FluidContext.hpp"
#include "OfflineClient.hpp"
#include "ParameterConstraints.hpp"
#include "ParameterSet.hpp"
#include "ParameterTypes.hpp"
#include "Result.hpp"
#include "TupleUtilities.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMeta.hpp"
#include <tuple>

namespace fluid {
namespace client {

enum ProcessState { kNoProcess, kProcessing, kDone, kDoneStillProcessing };

template <typename ParamType, ParamType& PD>
class FluidBaseClient //<const Tuple<Ts...>>
{
public:
  using ParamDescType = ParamType;
  using ParamSetType = ParameterSet<ParamDescType>;
  using ParamSetViewType = ParameterSetView<ParamDescType>;

  FluidBaseClient(ParamSetViewType& p) : mParams(std::ref(p)) {}

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  template <size_t N, typename T>
  void set(T&& x, Result* reportage) noexcept
  {
    mParams.template set(std::forward<T>(x), reportage);
  }

  index audioChannelsIn() const noexcept { return mAudioChannelsIn; }
  index audioChannelsOut() const noexcept { return mAudioChannelsOut; }
  index controlChannelsIn() const noexcept { return mControlChannelsIn; }
  index controlChannelsOut() const noexcept { return mControlChannelsOut; }

  index maxControlChannelsOut() const noexcept
  {
    return mMaxControlChannelsOut;
  }
  bool controlTrigger() const noexcept { return mControlTrigger; }

  index audioBuffersIn() const noexcept { return mBuffersIn; }
  index audioBuffersOut() const noexcept { return mBuffersOut; }

  constexpr static ParamDescType& getParameterDescriptors() { return PD; }

  double sampleRate() const noexcept { return mSampleRate; };
  void   sampleRate(double sr) { mSampleRate = sr; }
  
  void setParams(ParamSetViewType& p) { mParams = p; }

protected:
  void audioChannelsIn(const index x) noexcept { mAudioChannelsIn = x; }
  void audioChannelsOut(const index x) noexcept { mAudioChannelsOut = x; }
  void controlChannelsIn(const index x) noexcept { mControlChannelsIn = x; }
  void controlChannelsOut(const index x) noexcept { mControlChannelsOut = x; }
  void maxControlChannelsOut(const index x) noexcept
  {
    mMaxControlChannelsOut = x;
  }

  void controlTrigger(const bool x) noexcept { mControlTrigger = x; }

  void audioBuffersIn(const index x) noexcept { mBuffersIn = x; }
  void audioBuffersOut(const index x) noexcept { mBuffersOut = x; }

  std::reference_wrapper<ParamSetViewType> mParams;

private:
  index mAudioChannelsIn = 0;
  index mAudioChannelsOut = 0;
  index mControlChannelsIn = 0;
  index mControlChannelsOut = 0;
  index mMaxControlChannelsOut = 0;
  bool   mControlTrigger{false};
  index mBuffersIn = 0;
  index mBuffersOut = 0;
  double mSampleRate = 0;
};

// Used by hosts for detecting client capabilities at compile time
template <class T>
using isNonRealTime = typename std::is_base_of<Offline, T>::type;
template <class T>
using isRealTime = std::integral_constant<bool, isAudio<T> || isControl<T>>;

} // namespace client
} // namespace fluid
