/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
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

// Tag type for DataModel clients (like KDTree and friends)
struct ModelObject
{};

enum ProcessState { kNoProcess, kProcessing, kDone, kDoneStillProcessing };

struct ControlChannel {
  index count{0}; 
  index size{-1};
  index max{-1}; 
}; 

class FluidBaseClient
{
public:
  static constexpr auto& getParameterDescriptors() { return NoParameters; }

  index audioChannelsIn() const noexcept { return mAudioChannelsIn; }
  index audioChannelsOut() const noexcept { return mAudioChannelsOut; }
  index controlChannelsIn() const noexcept { return mControlChannelsIn; }
  ControlChannel controlChannelsOut() const noexcept { return mControlChannelsOut; }
  index maxControlChannelsOut() const noexcept
  {
    return mControlChannelsOut.max > -1 ? mControlChannelsOut.max : mControlChannelsOut.size;
  }
  bool   controlTrigger() const noexcept { return mControlTrigger; }
  index  audioBuffersIn() const noexcept { return mBuffersIn; }
  index  audioBuffersOut() const noexcept { return mBuffersOut; }
  double sampleRate() const noexcept { return mSampleRate; };
  void   sampleRate(double sr) { mSampleRate = sr; }

  const char* getInputLabel(index i) const
  {
    return i < asSigned(mInputLabels.size()) ? mInputLabels[asUnsigned(i)] : "";
  }

  const char* getOutputLabel(index i) const
  {
    return i < asSigned(mOutputLabels.size()) ? mOutputLabels[asUnsigned(i)] : "";
  }


protected:
  void audioChannelsIn(const index x) noexcept { mAudioChannelsIn = x; }
  void audioChannelsOut(const index x) noexcept { mAudioChannelsOut = x; }
  void controlChannelsIn(const index x) noexcept { mControlChannelsIn = x; }
  void controlChannelsOut(const ControlChannel x) noexcept { mControlChannelsOut = x; }
  void controlTrigger(const bool x) noexcept { mControlTrigger = x; }
  void audioBuffersIn(const index x) noexcept { mBuffersIn = x; }
  void audioBuffersOut(const index x) noexcept { mBuffersOut = x; }

  void setInputLabels(std::initializer_list<const char*> labels)
  {
    mInputLabels.clear();
    mInputLabels.reserve(labels.size());
    mInputLabels.insert(mInputLabels.end(),labels.begin(), labels.end());
  }

  void setOutputLabels(std::initializer_list<const char*> labels)
  {
    mOutputLabels.clear();
    mOutputLabels.reserve(labels.size());
    mOutputLabels.insert(mOutputLabels.end(),labels.begin(), labels.end());
  }

private:
  index  mAudioChannelsIn = 0;
  index  mAudioChannelsOut = 0;
  index  mControlChannelsIn = 0;
  ControlChannel  mControlChannelsOut {0,0,-1};
  bool   mControlTrigger{false};
  index  mBuffersIn = 0;
  index  mBuffersOut = 0;
  double mSampleRate = 0;
  std::vector<const char*> mInputLabels;
  std::vector<const char*> mOutputLabels;
};

struct AnalysisSize
{
  index window;
  index hop;    
}; 

template <typename C>
class ClientWrapper
{
public:
  using Client = C;
  using isNonRealTime = typename std::is_base_of<Offline, Client>::type;
  using isRealTime =
      std::integral_constant<bool, isAudio<Client> || isControl<Client>>;
  using isModelObject = typename std::is_base_of<ModelObject, Client>::type;

  template <typename T>
  using ParamDescTypeTest = typename T::ParamDescType;

  template <typename T>
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

  ClientWrapper(ParamSetViewType& p, FluidContext c) : mParams{p}, mClient{p, c} {}

  ClientWrapper(ClientWrapper&& x)
      : mParams{x.mParams}, mClient{std::move(x.mClient)}
  {
    mClient.setParams(mParams);
  }

  ClientWrapper& operator=(ClientWrapper&& x)
  {
    using std::swap;
    swap(mClient, x.mClient);
    mParams = x.mParams;
    mClient.setParams(mParams);
    return *this;
  }

  const Client& client() const { return mClient; }

  void reset(FluidContext& c) { mClient.reset(c); }

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

  AnalysisSize analysisSettings() { return mClient.analysisSettings(); }

  template <size_t N, typename T>
  void set(T&& x, Result* reportage) noexcept
  {
    mParams.template set(std::forward<T>(x), reportage);
  }

  auto latency() const { return mClient.latency(); }

  index audioChannelsIn() const noexcept { return mClient.audioChannelsIn(); }
  index audioChannelsOut() const noexcept { return mClient.audioChannelsOut(); }
  index controlChannelsIn() const noexcept
  {
    return mClient.controlChannelsIn();
  }
  ControlChannel controlChannelsOut() const noexcept
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

  const char* getInputLabel(index i) const
  {
    return mClient.getInputLabel(i);
  }

  const char* getOutputLabel(index i) const
  {
    return mClient.getOutputLabel(i);
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
using isRealTime = std::integral_constant<bool, isAudio<T> || isControlOut<T>>;

template <typename T>
class SharedClientRef; // forward declaration


template <typename T>
using IsSharedClient = isSpecialization<T, SharedClientRef>;

} // namespace client
} // namespace fluid
