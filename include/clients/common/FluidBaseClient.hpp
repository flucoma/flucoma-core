#pragma once

#include "AudioClient.hpp"
#include "FluidContext.hpp"
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


enum ProcessState { kNoProcess, kProcessing, kDone, kDoneStillProcessing };

class FluidBaseClient
{
public:
  size_t audioChannelsIn() const noexcept { return mAudioChannelsIn; }
  size_t audioChannelsOut() const noexcept { return mAudioChannelsOut; }
  size_t controlChannelsIn() const noexcept { return mControlChannelsIn; }
  size_t controlChannelsOut() const noexcept { return mControlChannelsOut; }
  size_t maxControlChannelsOut() const noexcept { return mMaxControlChannelsOut; }
  bool   controlTrigger() const noexcept { return mControlTrigger; }
  size_t audioBuffersIn() const noexcept { return mBuffersIn; }
  size_t audioBuffersOut() const noexcept { return mBuffersOut; }
  double sampleRate() const noexcept { return mSampleRate; };
  void  sampleRate(double sr) { mSampleRate = sr; }
protected:
  void audioChannelsIn(const size_t x) noexcept { mAudioChannelsIn = x; }
  void audioChannelsOut(const size_t x) noexcept { mAudioChannelsOut = x; }
  void controlChannelsIn(const size_t x) noexcept { mControlChannelsIn = x; }
  void controlChannelsOut(const size_t x) noexcept { mControlChannelsOut = x; }
  void maxControlChannelsOut(const size_t x) noexcept { mMaxControlChannelsOut = x; }
  void controlTrigger(const bool x) noexcept { mControlTrigger = x; }
  void audioBuffersIn(const size_t x) noexcept { mBuffersIn = x; }
  void audioBuffersOut(const size_t x) noexcept { mBuffersOut = x; }
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

template<typename C>
class ClientWrapper
{
 public:
  using Client = C;
  using isNonRealTime = typename std::is_base_of<Offline, Client>::type;
  using isRealTime = std::integral_constant<bool, isAudio<Client> || isControl<Client>>;


  template<typename T>
  using ParamDescTypeTest = typename T::ParamDescType;

  template<typename T>
  using MessageTypeTest = typename T::MessageType;

  using ParamDescType = typename DetectedOr<decltype(NoParameters),ParamDescTypeTest,Client>::type;
  using ParamSetType = ParameterSet<ParamDescType>;
  using ParamSetViewType = ParameterSetView<ParamDescType>;

  using MessageSetType = typename DetectedOr<decltype(NoMessages), MessageTypeTest,Client>::type;

  using HasParams = isDetected<ParamDescTypeTest, Client>;
  using HasMessages = isDetected<MessageTypeTest, Client>;
  
  
  template<typename P = HasParams>
  constexpr static auto  getParameterDescriptors() -> std::enable_if_t<P::value,ParamDescType>
  { return Client::getParameterDescriptors(); }
  
  template<typename P = HasParams>
  constexpr static auto getParameterDescriptors() -> std::enable_if_t<!P::value,ParamDescType>
  { return NoParameters; }
  
  template<typename M = HasMessages>
  constexpr static auto getMessageDescriptors() -> std::enable_if_t<M::value,MessageSetType>
  { return Client::getMessageDescriptors(); }
  
  template<typename M = HasMessages>
  constexpr static auto getMessageDescriptors() -> std::enable_if_t<!M::value,MessageSetType>
  { return NoMessages; }
  
  ClientWrapper(ParamSetViewType& p):mClient{p},mParams{p} {}
  
  const Client& client() const { return mClient; }
  
  template<typename T, typename Context>
  Result process(Context& c)
  {
      return mClient.template process<T>(c);
  }
  
  template <typename Input, typename Output>
  void process(Input input, Output output, FluidContext& c, bool reset = false) {
      mClient.process(input,output,c,reset);
  }
  
  template<size_t N,typename T,typename...Args>
  decltype(auto) invoke(T&, Args&&...args)
  {
    return getMessageDescriptors().template invoke<N>(mClient,std::forward<Args>(args)...);
  }

  size_t controlRate() { return mClient.controlRate(); }

  template <size_t N, typename T>
  void set(T &&x, Result *reportage) noexcept
  {
    mParams.template set(std::forward<T>(x), reportage);
  }

  auto latency() { return mClient.latency(); }

  size_t audioChannelsIn() const noexcept { return mClient.audioChannelsIn(); }
  size_t audioChannelsOut() const noexcept { return mClient.audioChannelsOut(); }
  size_t controlChannelsIn() const noexcept { return mClient.controlChannelsIn(); }
  size_t controlChannelsOut() const noexcept { return mClient.controlChannelsOut(); }

  size_t maxControlChannelsOut() const noexcept { return mClient.maxControlChannelsOut(); }
  bool   controlTrigger() const noexcept { return mClient.controlTrigger(); }

  size_t audioBuffersIn() const noexcept { return mClient.buffersIn(); }
  size_t audioBuffersOut() const noexcept { return mClient.buffersOut; }

  double sampleRate() const noexcept { return mClient.sampleRate(); };
  void  sampleRate(double sr) { mClient.sampleRate(sr); }
  
  template<typename P = HasParams>
  typename std::enable_if_t<P::value,void> setParams(ParamSetViewType& p)
  {
    mParams = p;
    mClient.setParams(p);
  }
  
  template<typename P = HasParams>
  typename std::enable_if_t<!P::value,void> setParams(ParamSetViewType& p) { mParams = p; }
  
private:
  Client mClient;
  std::reference_wrapper<ParamSetViewType> mParams;
};


// Used by hosts for detecting client capabilities at compile time
template <class T>
using isNonRealTime = typename std::is_base_of<Offline, T>::type;
template <class T>
using isRealTime = std::integral_constant<bool, isAudio<T> || isControl<T>>;

} // namespace client
} // namespace fluid
