/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "AudioClient.hpp"
#include "FluidContext.hpp"
#include "MessageSet.hpp"
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

//Tag type for DataModel clients (like KDTree and friends)
struct ModelObject{};

enum ProcessState { kNoProcess, kProcessing, kDone, kDoneStillProcessing };

class FluidBaseClient
{
public:

  static constexpr auto& getParameterDescriptors() { return NoParameters; }

  index audioChannelsIn() const noexcept { return mAudioChannelsIn; }
  index audioChannelsOut() const noexcept { return mAudioChannelsOut; }
  index controlChannelsIn() const noexcept { return mControlChannelsIn; }
  index controlChannelsOut() const noexcept { return mControlChannelsOut; }
  index maxControlChannelsOut() const noexcept
  {
    return mMaxControlChannelsOut;
  }
  bool   controlTrigger() const noexcept { return mControlTrigger; }
  index  audioBuffersIn() const noexcept { return mBuffersIn; }
  index  audioBuffersOut() const noexcept { return mBuffersOut; }
  double sampleRate() const noexcept { return mSampleRate; };
  void   sampleRate(double sr) { mSampleRate = sr; }

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

private:
  index  mAudioChannelsIn = 0;
  index  mAudioChannelsOut = 0;
  index  mControlChannelsIn = 0;
  index  mControlChannelsOut = 0;
  index  mMaxControlChannelsOut = 0;
  bool   mControlTrigger{false};
  index  mBuffersIn = 0;
  index  mBuffersOut = 0;
  double mSampleRate = 0;
};

template <typename C>
class ClientWrapper
{
public:
  using Client = C;
  using isNonRealTime = typename std::is_base_of<Offline, Client>::type;
  using isRealTime =
      std::integral_constant<bool, isAudio<Client> || isControl<Client>>;
  using isModelObject = typename std::is_base_of<ModelObject,Client>::type; 

  template <typename T>
  using ParamDescTypeTest = typename T::ParamDescType;

  template<typename T> 
  using MessageTypeTest = decltype(T::getMessageDescriptors()); 

  using ParamDescType = typename DetectedOr<decltype(NoParameters),
                                            ParamDescTypeTest, Client>::type;
  using ParamSetType = ParameterSet<ParamDescType>;
  using ParamSetViewType = ParameterSetView<ParamDescType>;

  using MessageSetType =
      typename DetectedOr<decltype(NoMessages), MessageTypeTest, Client>::type;

  using HasParams = isDetected<ParamDescTypeTest, Client>;
  using HasMessages = isDetected<MessageTypeTest, Client>;

  constexpr static ParamDescType descript = Client::getParameterDescriptors();
  
  template <typename P = HasParams>
  constexpr static auto getParameterDescriptors()
      -> std::enable_if_t<P::value, ParamDescType&>
  {
//    return Client::getParameterDescriptors();

     return descript;
  }

  template <typename P = HasParams>
  constexpr static auto getParameterDescriptors()
      -> std::enable_if_t<!P::value, ParamDescType&>
  {
    return NoParameters;
  }

  template <typename M = HasMessages>
  constexpr static auto getMessageDescriptors()
      -> std::enable_if_t<M::value, MessageSetType>
  {
    return Client::getMessageDescriptors();
  }

  template <typename M = HasMessages>
  constexpr static auto getMessageDescriptors()
      -> std::enable_if_t<!M::value, MessageSetType>
  {
    return NoMessages;
  }

  ClientWrapper(ParamSetViewType& p) : mParams{p} ,mClient{p} {}

  ClientWrapper(ClientWrapper&& x):
      mParams{x.mParams},
      mClient{std::move(x.mClient)}
  {
    mClient.setParams(mParams);
  }

  ClientWrapper& operator=(ClientWrapper&& x)
  {
    using std::swap;
    swap(mClient,x.mClient);
    mParams = x.mParams;
    mClient.setParams(mParams);
    return *this;
  }

  const Client& client() const { return mClient; }

  void reset() { mClient.reset(); }

  template <typename T, typename Context>
  Result process(Context& c)
  {
    return mClient.template process<T>(c);
  }

  template <typename Input, typename Output>
  void process(Input& input, Output& output, FluidContext& c)
  {
    mClient.process(input, output, c);
  }

  template <size_t N, typename T, typename... Args>
  decltype(auto) invoke(T&, Args&&... args)
  {
    return getMessageDescriptors().template invoke<N>(
        mClient, std::forward<Args>(args)...);
  }

  index controlRate() { return mClient.controlRate(); }

  template <size_t N, typename T>
  void set(T&& x, Result* reportage) noexcept
  {
    mParams.template set(std::forward<T>(x), reportage);
  }

  auto latency() { return mClient.latency(); }

  index audioChannelsIn() const noexcept { return mClient.audioChannelsIn(); }
  index audioChannelsOut() const noexcept { return mClient.audioChannelsOut(); }
  index controlChannelsIn() const noexcept
  {
    return mClient.controlChannelsIn();
  }
  index controlChannelsOut() const noexcept
  {
    return mClient.controlChannelsOut();
  }

  index maxControlChannelsOut() const noexcept
  {
    return mClient.maxControlChannelsOut();
  }
  bool controlTrigger() const noexcept { return mClient.controlTrigger(); }

  index audioBuffersIn() const noexcept { return mClient.buffersIn(); }
  index audioBuffersOut() const noexcept { return mClient.buffersOut; }

  double sampleRate() const noexcept { return mClient.sampleRate(); };
  void   sampleRate(double sr) { mClient.sampleRate(sr); }

  template <typename P = HasParams>
  typename std::enable_if_t<P::value, void> setParams(ParamSetViewType& p)
  {
    mParams = p;
    mClient.setParams(p);
  }

  template <typename P = HasParams>
  typename std::enable_if_t<!P::value, void> setParams(ParamSetViewType& p)
  {
    mParams = p;
  }

private:
  std::reference_wrapper<ParamSetViewType> mParams;
  
  Client mClient;
};


template <typename C>
constexpr typename ClientWrapper<C>::ParamDescType ClientWrapper<C>::descript;


// Used by hosts for detecting client capabilities at compile time
template <class T>
using isNonRealTime = typename std::is_base_of<Offline, T>::type;
template <class T>
using isRealTime = std::integral_constant<bool, isAudio<T> || isControl<T>>;

template <typename T>
class SharedClientRef; // forward declaration


template <typename T>
using IsSharedClient = isSpecialization<T, SharedClientRef>;

} // namespace client
} // namespace fluid
