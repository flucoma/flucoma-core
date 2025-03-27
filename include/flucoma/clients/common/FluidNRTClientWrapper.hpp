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

#include "../common/BufferAdaptor.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/MemoryBufferAdaptor.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/SpikesToTimes.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <deque>
#include <future>
#include <thread>
#include <vector>

namespace fluid {
namespace client {

namespace impl {
//////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename B>
auto constexpr makeWrapperInputs(B b)
{
  return defineParameters(std::forward<B>(b),
                          LongParam("startFrame", "Source Offset", 0, Min(0)),
                          LongParam("numFrames", "Number of Frames", -1),
                          LongParam("startChan", "Start Channel", 0, Min(0)),
                          LongParam("numChans", "Number of Channels", -1));
}

template <typename B>
auto constexpr makeWrapperInputs(B b1, B b2)
{
  return defineParameters(
      std::forward<B>(b1),
      LongParam("startFrameA", "Source A Offset", 0, Min(0)),
      LongParam("numFramesA", "Source A Number of Frames", -1),
      LongParam("startChanA", "Source A Start Channel", 0, Min(0)),
      LongParam("numChansA", "Source A Number of Channels", -1),
      std::forward<B>(b2),
      LongParam("startFrameB", "Source B Offset", 0, Min(0)),
      LongParam("numFramesB", "Source B Number of Frames", -1),
      LongParam("startChanB", "Source B Start Channel", 0, Min(0)),
      LongParam("numChansB", "Source B Number of Channels", -1));
}

template <typename... B>
auto constexpr makeWrapperOutputs(B... b)
{
  return defineParameters(std::forward<B>(b)...);
}

template <typename T, size_t N, size_t... Is>
auto constexpr spitIns(T (&a)[N], std::index_sequence<Is...>)
{
  return makeWrapperInputs(std::forward<T>(a[Is])...);
}

template <typename T, size_t N, size_t... Is>
auto constexpr spitOuts(T (&a)[N], std::index_sequence<Is...>)
{
  return makeWrapperOutputs(std::forward<T>(a[Is])...);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
using InputBufferSpec = ParamSpec<InputBufferT, Fixed<false>>;
using BufferSpec = ParamSpec<BufferT, Fixed<false>>;
//////////////////////////////////////////////////////////////////////////////////////////////////////

template <size_t... Is, size_t... Js, typename... Ts, typename... Us>
constexpr auto joinParameterDescriptors(
    ParameterDescriptorSet<std::index_sequence<Is...>, std::tuple<Ts...>> x,
    ParameterDescriptorSet<std::index_sequence<Js...>, std::tuple<Us...>> y)
{
  return ParameterDescriptorSet<
      typename JoinOffsetSequence<std::index_sequence<Is...>,
                                  std::index_sequence<Js...>>::type,
      std::tuple<Ts..., Us...>>{
      std::tuple_cat(x.descriptors(), y.descriptors())};
}

template <class ParamDesc>
struct ParamsSize;

template <class Offsets, class Tuple>
struct ParamsSize<const ParameterDescriptorSet<Offsets, Tuple>>
{
  static constexpr size_t value = std::tuple_size<Tuple>::value;
};

struct IsFFTParam
{
  template <typename T>
  using apply =
      std::is_same<FFTParamsT, typename std::tuple_element<0, T>::type>;
};

template <typename T>
struct IsControlOut
{
  constexpr static bool value = std::is_base_of<ControlOut, T>::value;
};

template <typename T>
struct IsControlOut<ClientWrapper<T>>
{
  constexpr static bool value{std::is_base_of<ControlOut, T>::value};
};


template <class RTClient>
struct AddPadding
{
  using Index = typename RTClient::ParamDescType::IndexList;
  using Types = typename RTClient::ParamDescType::DescriptorType;
  static constexpr bool HasFFT =
      impl::FilterTupleIndices<IsFFTParam, Types, Index>::type::size() > 0;
  static constexpr bool HasControlOut = IsControlOut<RTClient>::value;
  //  static constexpr size_t value = HasControlOut? 2 : 1;

  static constexpr size_t value = HasFFT && HasControlOut    ? 2
                                  : HasFFT && !HasControlOut ? 1
                                  : !HasFFT && HasControlOut ? 3
                                                             : 0;


  //  static constexpr size_t value = std::conditional_t<HasFFT,
  //      std::conditional_t<HasControlOut, std::integral_constant<size_t, 2>,
  //                         std::integral_constant<size_t, 1>>,
  //      std::integral_constant<size_t, 0>>()();
};

// Special case for Loudness :`-(
template <typename C>
using NonFFTWithControlOut = std::enable_if_t<AddPadding<C>::value == 3>;

template <typename C>
using FFTWithControlOut = std::enable_if_t<AddPadding<C>::value == 2>;
template <typename C>
using FFTWithAudioOut = std::enable_if_t<AddPadding<C>::value == 1>;
template <typename C>
using NonFFT = std::enable_if_t<AddPadding<C>::value == 0>;

//////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename HostMatrix, typename HostVectorView>
struct StreamingControl;

template <template <typename, typename> class AdaptorType, class Client,
          typename ParamType, ParamType& PD, size_t Ins, size_t Outs>
class NRTClientWrapper : public OfflineIn, public OfflineOut
{
public:
  // Host buffers are always float32 (?)
  using RTClient = ClientWrapper<Client>;
  using HostVector = FluidTensor<float, 1>;
  using HostMatrix = FluidTensor<float, 2>;
  using HostVectorView = FluidTensorView<float, 1>;
  using HostMatrixView = FluidTensorView<float, 2>;
  static constexpr auto isControl =
      std::is_same<AdaptorType<HostMatrix, HostVectorView>,
                   StreamingControl<HostMatrix, HostVectorView>>();
  
  using ParamDescType = ParamType;
  using ParamSetType = ParameterSet<ParamDescType>;
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  using RTParamDescType = typename RTClient::ParamDescType;
  using RTParamSetViewType = ParameterSetView<typename RTClient::ParamDescType>;

  using MessageSetType = typename RTClient::MessageSetType;

  constexpr static ParamDescType& getParameterDescriptors() { return PD; }

  using isRealTime = std::false_type;
  using isNonRealTime = std::true_type;

  // The client will be accessing its parameter by a bunch of indices that need
  // ofsetting now
  //  using Client =
  //  RTClient<impl::ParameterSet_Offset<Params,ParamOffset>,T,U>;
  // None of that for outputs though

  static constexpr size_t ParamOffset =
      ParamsSize<ParamDescType>::value - ParamsSize<RTParamDescType>::value;

  static constexpr index VectorSize = 64;

  using WrappedClient = RTClient;

  using isModelObject = typename RTClient::isModelObject;

  static auto getMessageDescriptors()
  {
    return RTClient::getMessageDescriptors();
  }

  NRTClientWrapper(ParamSetViewType& p, FluidContext)
      : mParams{p}, mNRTContext{VectorSize, FluidDefaultAllocator()},
        mRealTimeParams{RTClient::getParameterDescriptors(),
                        mParams.get().template subset<ParamOffset>()},
        mClient{mRealTimeParams, mNRTContext}
  {}

  NRTClientWrapper(const NRTClientWrapper& x)
      : mParams{x.mParams}, mNRTContext{x.mNRTContext},
        mRealTimeParams{RTClient::getParameterDescriptors(),
                        mParams.get().template subset<ParamOffset>()},
        mClient{mRealTimeParams, mNRTContext}
  {}


  NRTClientWrapper(NRTClientWrapper&& x)
      : mParams{std::move(x.mParams)},
        mNRTContext{std::move(x.mNRTContext)}, mClient{std::move(x.mClient)}
  {
    mRealTimeParams =
        RTParamSetViewType(RTClient::getParameterDescriptors(),
                           mParams.get().template subset<ParamOffset>());
    mClient.setParams(mRealTimeParams);
  }

  NRTClientWrapper& operator=(const NRTClientWrapper& x)
  {
    mParams = x.mParams;
    mClient = x.mClient;
    mNRTContext = x.mNRTContext;
    mRealTimeParams =
        RTParamSetViewType(RTClient::getParameterDescriptors(),
                           mParams.get().template subset<ParamOffset>());
    mClient.setParams(mRealTimeParams);
    return *this;
  }

  NRTClientWrapper& operator=(NRTClientWrapper&& x)
  {
    using std::swap;
    swap(mClient, x.mClient);
    swap(mParams, x.mParams);
    swap(mNRTContext, x.mNRTContext);
    mRealTimeParams =
        RTParamSetViewType(RTClient::getParameterDescriptors(),
                           mParams.get().template subset<ParamOffset>());
    mClient.setParams(mRealTimeParams);
    return *this;
  }


  template <std::size_t N>
  auto& get() noexcept
  {
    return mParams.get().template get<N>();
  }
  //  template <std::size_t N> bool changed() noexcept { return mParams.template
  //  changed<N>(); }

  index          audioChannelsIn() const noexcept { return 0; }
  index          audioChannelsOut() const noexcept { return 0; }
  index          controlChannelsIn() const noexcept { return 0; }
  ControlChannel controlChannelsOut() const noexcept { return {0, 0}; }
  /// Map delegate audio / control channels to audio buffers
  index audioBuffersIn() const noexcept { return mClient.audioChannelsIn(); }
  index audioBuffersOut() const noexcept
  {
    return isControl ? 1 : mClient.audioChannelsOut();
  }

  void setParams(ParamSetViewType& p)
  {
    mParams = p;
    mRealTimeParams =
        RTParamSetViewType(RTClient::getParameterDescriptors(),
                           mParams.get().template subset<ParamOffset>());
    mClient.setParams(mRealTimeParams);
  }

  template <size_t N, typename... Args>
  decltype(auto) invoke(Args&&... args)
  {
    return invokeDelegate<N>(std::forward<Args>(args)...);
  }

  template <typename T>
  Result process(FluidContext& c)
  {
    auto constexpr inputCounter = std::make_index_sequence<Ins>();
    auto constexpr outputCounter = std::make_index_sequence<Outs>();

    auto inputBuffers = fetchInputBuffers(inputCounter);
    auto outputBuffers = fetchOutputBuffers(outputCounter);

    std::array<index, Ins> inFrames;
    std::array<index, Ins> inChans;

    // check buffers exist
    index count = 0;
    for (auto&& b : inputBuffers)
    {

      index requestedFrames = b.nFrames;
      index requestedChans = b.nChans;

      auto rangeCheck = bufferRangeCheck(
          b.buffer, b.startFrame, requestedFrames, b.startChan, requestedChans);

      if (!rangeCheck.ok()) return rangeCheck;

      inFrames[asUnsigned(count)] = requestedFrames;
      inChans[asUnsigned(count)] = requestedChans;
      mClient.sampleRate(BufferAdaptor::ReadAccess(b.buffer).sampleRate());
      count++;
    }

    if (std::all_of(outputBuffers.begin(), outputBuffers.end(), [](auto& b) {
          if (!b) return true;

          BufferAdaptor::Access buf(b);
          return !buf.exists();
        }))
      return {Result::Status::kError, "No valid output has been set"}; // error


    Result r{Result::Status::kOk, ""};

    // Remove non-existent output buffers from the output buffers vector, so
    // clients don't try and use them
    std::transform(
        outputBuffers.begin(), outputBuffers.end(), outputBuffers.begin(),
        [&r](auto& b) -> BufferAdaptor* {
          if (!b) return nullptr;
          BufferAdaptor::Access buf(b);
          if (!buf.exists())
          {
            r.set(Result::Status::kWarning);
            r.addMessage("One or more of your output buffers doesn't exist\n");
          }
          return buf.exists() ? b : nullptr;
        });


    index numFrames = *std::min_element(inFrames.begin(), inFrames.end());
    index numChannels = *std::min_element(inChans.begin(), inChans.end());

    mNRTContext.task(c.task());

    Result processResult = AdaptorType<HostMatrix, HostVectorView>::process(
        mClient, inputBuffers, outputBuffers, numFrames, numChannels,
        userPadding<>(), mNRTContext);

    if (!processResult.ok())
    {
      r.set(processResult.status());
      r.addMessage(processResult.message());
    }

    return r;
  }

  template <typename C = RTClient>
  std::pair<index, index> userPadding(FFTWithControlOut<C>* = 0)
  {
    using FFTLookup = std::reference_wrapper<FFTParams>;
    index userPaddingParamValue = get<ParamOffset - 1>();
    auto  fftSettings = mParams.get().template get<FFTLookup>();
    return {FFTParams::padding(fftSettings, userPaddingParamValue),
            userPaddingParamValue};
  }

  template <typename C = RTClient>
  std::pair<index, index> userPadding(FFTWithAudioOut<C>* = 0)
  {
    using FFTLookup = std::reference_wrapper<FFTParams>;
    index winSize = mParams.get().template get<FFTLookup>().winSize();
    return {winSize >> 1, 1};
  }

  template <typename C = RTClient>
  std::pair<index, index> userPadding(NonFFT<C>* = 0)
  {
    return {0, 0};
  }

  // FIXME: This has to rely on the specific structure of LoudnessCLient at the
  // moment, which is bad, bad, bad
  template <typename C = RTClient>
  std::pair<index, index> userPadding(NonFFTWithControlOut<C>* = 0)
  {
    //    using FFTLookup = std::reference_wrapper<FFTParams>;
    //    index userPaddingParamValue = get<ParamOffset - 1>();
    //    auto  fftSettings = mParams.get().template get<FFTLookup>();
    //    return {FFTParams::padding(fftSettings,
    //    userPaddingParamValue),userPaddingParamValue};
    index            userPaddingParamValue = get<ParamOffset - 1>();
    constexpr size_t WinOffset = 3;
    constexpr size_t HopOffset = 4;

    assert(
        (mParams.get().template descriptorAt<ParamOffset + WinOffset>()).name ==
        "windowSize");

    index winSize = get<ParamOffset + WinOffset>();
    index hopSize = get<ParamOffset + HopOffset>();
    using Op = index (*)(index, index);
    static std::array<Op, 3> options{
        [](index, index) -> index { return 0; },
        [](index win, index) { return win >> 1; },
        [](index win, index hop) { return win - hop; }};
    return {options[asUnsigned(userPaddingParamValue)](winSize, hopSize),
            userPaddingParamValue};
  }


private:
  template <size_t N, typename T>
  struct GetWinSize
  {
    void operator()(typename T::type& param, index& x) { x = param.winSize(); }
  };

  template <size_t I>
  BufferProcessSpec fetchInputBuffer()
  {
    return {get<I>().get(), get<I + 1>(), get<I + 2>(), get<I + 3>(),
            get<I + 4>()};
  }

  template <size_t... Is>
  std::array<BufferProcessSpec, sizeof...(Is)>
  fetchInputBuffers(std::index_sequence<Is...>)
  {
    return {{fetchInputBuffer<Is * 5>()...}};
  }

  template <size_t... Is>
  std::array<BufferAdaptor*, sizeof...(Is)>
  fetchOutputBuffers(std::index_sequence<Is...>)
  {
    return {{get<Is + (Ins * 5)>().get()...}};
  }


  // We need to delegate invoke calls for message back to the actual RT client,
  // where the actual functions live: peel off the 1st client argument, and
  // replace with our client instance
  template <size_t N, typename NRTClient, typename... Args>
  decltype(auto) invokeDelegate(NRTClient&, Args&&... args)
  {
    return mClient.template invoke<N>(mClient, std::forward<Args>(args)...);
  }

  std::reference_wrapper<ParamSetViewType> mParams;
  FluidContext                             mNRTContext;
  RTParamSetViewType                       mRealTimeParams;
  WrappedClient                            mClient;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename HostMatrix, typename HostVectorView>
struct Streaming
{
  template <typename Client, typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,
                        OutputList& outputBuffers, index nFrames, index nChans,
                        std::pair<index, index> userPadding, FluidContext& c)
  {
    // To account for process latency we need to copy the buffers with padding
    std::vector<HostMatrix> outputData;
    std::vector<HostMatrix> inputData;
    index VectorSize = c.hostVectorSize(); // todo sort this aht
    outputData.reserve(outputBuffers.size());
    inputData.reserve(inputBuffers.size());

    index startPadding = client.latency() + userPadding.first;
    index totalPadding = startPadding + userPadding.first;

    // store a whole multiple of VectorSize's worth of data to get padding at
    // far end
    index nHops = static_cast<index>(
        std::ceil(double(nFrames + totalPadding) / VectorSize));

    std::fill_n(std::back_inserter(outputData), outputBuffers.size(),
                HostMatrix(nChans, VectorSize * nHops));
    std::fill_n(std::back_inserter(inputData), inputBuffers.size(),
                HostMatrix(nChans, VectorSize * nHops));

    double sampleRate{0};

    // copy input data
    for (index i = 0; i < nChans; ++i)
    {
      for (index j = 0; j < asSigned(inputBuffers.size()); ++j)
      {
        BufferAdaptor::ReadAccess thisInput(inputBuffers[asUnsigned(j)].buffer);
        inputData[asUnsigned(j)].row(i)(Slice(userPadding.first, nFrames)) <<=
            thisInput.samps(inputBuffers[asUnsigned(j)].startFrame, nFrames,
                            inputBuffers[asUnsigned(j)].startChan + i);
        if (i == 0 && j == 0) sampleRate = thisInput.sampleRate();
      }
    }

    std::vector<HostVectorView> inputs(inputBuffers.size(), {nullptr, 0, 0});
    std::vector<HostVectorView> outputs(outputBuffers.size(), {nullptr, 0, 0});
    //    inputs.reserve(inputBuffers.size());

    for (index i = 0; i < nChans; ++i)
    {
      if (c.task())
        c.task()->iterationUpdate(static_cast<double>(i),
                                  static_cast<double>(nChans));

      client.reset(c);

      for (index j = 0; j < nHops; ++j)
      {
        for (std::size_t k = 0; k < inputBuffers.size(); ++k)
          inputs[k] = inputData[k].row(i)(Slice(j * VectorSize, VectorSize));
        for (std::size_t k = 0; k < outputBuffers.size(); ++k)
          outputs[k] = outputData[k].row(i)(Slice(j * VectorSize, VectorSize));
        client.process(inputs, outputs, c);

        if (c.task() &&
            !c.task()->processUpdate(static_cast<double>(j + 1 + (nHops * i)),
                                     static_cast<double>(nHops * nChans)))
          break;
      }
    }

    for (index i = 0; i < asSigned(outputBuffers.size()); ++i)
    {
      if (!outputBuffers[asUnsigned(i)]) continue;
      BufferAdaptor::Access thisOutput(outputBuffers[asUnsigned(i)]);
      Result                r = thisOutput.resize(nFrames, nChans, sampleRate);
      if (!r.ok()) return r;
      for (index j = 0; j < nChans; ++j)
        thisOutput.samps(j) <<=
            outputData[asUnsigned(i)].row(j)(Slice(startPadding, nFrames));
    }

    return {};
  }
};
//////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename HostMatrix, typename HostVectorView>
struct StreamingControl
{
  template <typename Client, typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,
                        OutputList& outputBuffers, index nFrames, index nChans,
                        std::pair<index, index> userPadding, FluidContext& c)
  {
    // To account for process latency we need to copy the buffers with padding
    std::vector<HostMatrix> inputData;
    index                   maxFeatures = client.maxControlChannelsOut();

    inputData.reserve(inputBuffers.size());

    index startPadding = client.latency() + userPadding.first;

    index totalPadding = startPadding + userPadding.first;
    index controlRate = client.analysisSettings().hop;

    index paddedLength = nFrames + totalPadding;

    //    for descriptors in full padding mode, round analysis length up to a
    //    whole number of hops to ensure that last frame gets a fair showing
    if (userPadding.second == 2)
      paddedLength = static_cast<index>(
          std::ceil(double(paddedLength) / controlRate) * controlRate);

    index windowSize = client.analysisSettings().window;
    index nAnalysisFrames = static_cast<index>(
        1 + std::floor((paddedLength - windowSize) / controlRate));
    c.hostVectorSize(controlRate);
    std::fill_n(std::back_inserter(inputData), inputBuffers.size(),
                HostMatrix(nChans, paddedLength));

    std::vector<HostMatrix> outputData; 
    outputData.reserve(outputBuffers.size());
    std::fill_n(std::back_inserter(outputData), outputBuffers.size(),
                HostMatrix(nChans * maxFeatures, nAnalysisFrames));

    double     sampleRate{0};
    // Copy input data
    for (index i = 0; i < nChans; ++i)
    {
      for (index j = 0; j < asSigned(inputBuffers.size()); ++j)
      {
        BufferAdaptor::ReadAccess thisInput(inputBuffers[asUnsigned(j)].buffer);
        if (i == 0 && j == 0) sampleRate = thisInput.sampleRate();
        inputData[asUnsigned(j)].row(i)(Slice(userPadding.first, nFrames)) <<=
            thisInput.samps(inputBuffers[asUnsigned(j)].startFrame, nFrames,
                            inputBuffers[asUnsigned(j)].startChan + i);
      }
    }
    FluidTask* task = c.task();

    for (index i = 0; i < nChans; ++i)
    {
      client.reset(c);
      for (index j = 0; j < nAnalysisFrames; ++j)
      {
        index t = j * c.hostVectorSize();

        std::vector<HostVectorView> inputs;
        inputs.reserve(inputBuffers.size());
        std::vector<HostVectorView> outputs;
        outputs.reserve(outputBuffers.size());

        for (index k = 0; k < asSigned(inputBuffers.size()); ++k)
        {
          inputs.emplace_back(
              inputData[asUnsigned(k)].row(i)(Slice(t, controlRate)));
        }

        for(auto& out: outputData)
        {
          outputs.push_back(
            out.col(j)(Slice(i * maxFeatures, maxFeatures)));
        }

        client.process(inputs, outputs, c);

        if (task && !task->processUpdate(
                        static_cast<double>(j + 1 + (nAnalysisFrames * i)),
                        static_cast<double>(nAnalysisFrames * nChans)))
          break;
      }
    }

    for (auto outs = std::pair{outputBuffers.begin(), outputData.begin()};
         outs.first != outputBuffers.end(); ++outs.first, ++outs.second)
    {
      if (!(*outs.first)) continue;
      BufferAdaptor::Access thisOutput(*outs.first);
      index                 nFeatures = client.controlChannelsOut().size;
      index                 latencyHops = client.latency() / controlRate;
      index                 keepHops = nAnalysisFrames - latencyHops;

      Result resizeResult = thisOutput.resize(keepHops, nChans * nFeatures,
                                              sampleRate / controlRate);

      if (!resizeResult.ok()) return resizeResult;
      for (index i = 0; i < nFeatures; ++i)
      {
        for (index j = 0; j < nChans; ++j)
          thisOutput.samps(i + j * nFeatures) <<=
              outs.second->row(i + j * maxFeatures)(Slice(latencyHops, keepHops));
      }
    }
    
    return {};
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename HostMatrix, typename HostVectorView>
struct Slicing
{
  template <typename Client, typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,
                        OutputList& outputBuffers, index nFrames, index nChans,
                        std::pair<index, index> /*userPadding*/,
                        FluidContext& c)
  {

    assert(inputBuffers.size() == 1);
    assert(outputBuffers.size() == 1);

    index startPadding = client.latency();
    index totalPadding = startPadding;
    index VectorSize = c.hostVectorSize();
    index nHops = static_cast<index>(
        std::ceil(double(nFrames + totalPadding) / VectorSize));

    HostMatrix monoSource(1, nHops * VectorSize);

    BufferAdaptor::ReadAccess src(inputBuffers[0].buffer);
    // Make a mono sum;
    for (index i = inputBuffers[0].startChan;
         i < nChans + inputBuffers[0].startChan; ++i)
    {
      monoSource.row(0)(Slice(0, nFrames))
          .apply(src.samps(inputBuffers[0].startFrame, nFrames, i),
                 [](float& x, float y) { x += y; });
    }

    HostMatrix onsetPoints(1, nHops * VectorSize);

    std::vector<HostVectorView> input{{nullptr, 0, 0}};
    std::vector<HostVectorView> output{{nullptr, 0, 0}};
    client.reset(c);
    for (index i = 0, N = nHops; i < N; ++i)
    {
      input[0] = monoSource.row(0)(Slice(i * VectorSize, VectorSize));
      output[0] = onsetPoints.row(0)(Slice(i * VectorSize, VectorSize));

      client.process(input, output, c);
      if (c.task() && !c.task()->processUpdate(static_cast<double>(i),
                                               static_cast<double>(N)))
        break;
    }

    if (startPadding)
    {
      auto paddingAudio = onsetPoints(0, Slice(0, startPadding));
      auto numNegativeTimeOnsets =
          std::count_if(paddingAudio.begin(), paddingAudio.end(),
                        [](float x) { return x > 0; });

      if (numNegativeTimeOnsets > 0) onsetPoints(0, startPadding) = 1;
    }

    return impl::spikesToTimes(onsetPoints(0, Slice(startPadding, nFrames)),
                               outputBuffers[0], 1, inputBuffers[0].startFrame,
                               nFrames, src.sampleRate());
  }
};

} // namespace impl
//////////////////////////////////////////////////////////////////////////////////////////////////////

template <class RTClient, typename Params, Params& PD, index Ins, index Outs>
using NRTStreamAdaptor =
    impl::NRTClientWrapper<impl::Streaming, RTClient, Params, PD, Ins, Outs>;

template <class RTClient, typename Params, Params& PD, index Ins, index Outs>
using NRTSliceAdaptor =
    impl::NRTClientWrapper<impl::Slicing, RTClient, Params, PD, Ins, Outs>;

template <class RTClient, typename Params, Params& PD, index Ins, index Outs>
using NRTControlAdaptor =
    impl::NRTClientWrapper<impl::StreamingControl, RTClient, Params, PD, Ins,
                           Outs>;


//////////////////////////////////////////////////////////////////////////////////////////////////////

template <class RTClient, class... Outs>
auto constexpr addFFTPadding(Outs&&... outs,
                             std::enable_if_t <
                                 impl::AddPadding<RTClient>::value<2>* = 0)
{
  return defineParameters(std::forward<Outs>(outs)...);
}

template <class RTClient, class... Outs>
auto constexpr addFFTPadding(
    Outs&&... outs,
    std::enable_if_t<impl::AddPadding<RTClient>::value >= 2>* = 0)
{
  return defineParameters(
      std::forward<Outs>(outs)...,
      EnumParam("padding", "Added Padding", 1, "None", "Default", "Full"));
}

template <class RTClient, typename... Outs>
auto constexpr makeNRTParams(impl::InputBufferSpec&& in,
                             impl::InputBufferSpec&& in2, Outs&&... outs)
{
  return impl::joinParameterDescriptors(
      impl::joinParameterDescriptors(
          impl::makeWrapperInputs(in, in2),
          addFFTPadding<RTClient, Outs...>(std::forward<Outs>(outs)...)),
      ClientWrapper<RTClient>::getParameterDescriptors());
}

template <class RTClient, typename... Outs>
auto constexpr makeNRTParams(impl::InputBufferSpec&& in, Outs&&... outs)
{
  return impl::joinParameterDescriptors(
      impl::joinParameterDescriptors(
          impl::makeWrapperInputs(in),
          addFFTPadding<RTClient, Outs...>(std::forward<Outs>(outs)...)),
      ClientWrapper<RTClient>::getParameterDescriptors());
}


//////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NRTClient>
class NRTThreadingAdaptor : public OfflineIn, public OfflineOut
{
public:
  using Client = NRTClient;
  using ClientPointer = typename std::shared_ptr<NRTClient>;
  using ParamDescType = typename NRTClient::ParamDescType;
  using ParamSetType = typename NRTClient::ParamSetType;
  using ParamSetViewType = typename NRTClient::ParamSetViewType;
  using MessageSetType = typename NRTClient::MessageSetType;
  using isRealTime = std::false_type;
  using isNonRealTime = std::true_type;
  using isModelObject = typename NRTClient::isModelObject;

  constexpr static ParamDescType& getParameterDescriptors()
  {
    return NRTClient::getParameterDescriptors();
  }
  constexpr static auto getMessageDescriptors()
  {
    return NRTClient::getMessageDescriptors();
  }

  index          audioChannelsIn() const noexcept { return 0; }
  index          audioChannelsOut() const noexcept { return 0; }
  index          controlChannelsIn() const noexcept { return 0; }
  ControlChannel controlChannelsOut() const noexcept { return {0, 0}; }
  index          audioBuffersIn() const noexcept
  {
    return ParamDescType::template NumOf<InputBufferT>();
  }
  index audioBuffersOut() const noexcept
  {
    return ParamDescType::template NumOf<BufferT>();
  }

  void setParams(ParamSetType& p)
  {
    mHostParams = p;
    mClient->setParams(mHostParams);
  }

  NRTThreadingAdaptor(ParamSetType& p, FluidContext c)
      : mHostParams{p}, mClient{new NRTClient{mHostParams, c}}
  {}

  NRTThreadingAdaptor(NRTThreadingAdaptor&& x) : mHostParams{x.mHostParams}
  {
    swap(std::move(x), false);
  }

  NRTThreadingAdaptor& operator=(NRTThreadingAdaptor&& x)
  {
    swap(std::move(x), true);
    return *this;
  }

  ~NRTThreadingAdaptor()
  {

    mQueue.clear();

    if (mThreadedTask)
    {
      mThreadedTask->cancel(false);
      mThreadedTask->join();
      //      mThreadedTask.release();
    }
  }

  Result enqueue(ParamSetType& p, std::function<void()> callback = {})
  {
    if (mThreadedTask && (mSynchronous || !mQueueEnabled))
      return {Result::Status::kError, "already processing"};

    mQueue.push_back({p, callback});

    return {};
  }

  Result process()
  {
    if (mThreadedTask && (mSynchronous || !mQueueEnabled))
      return {Result::Status::kError, "already processing"};

    if (mThreadedTask) return Result();

    Result result;

    if (mQueue.empty())
      return {Result::Status::kWarning, "Process() called on empty queue"};

    if (mSynchronous) mSynchronousDone = false;

    mThreadedTask = std::unique_ptr<ThreadedTask>(
        new ThreadedTask(mClient, mQueue.front(), mSynchronous));
    mQueue.pop_front();

    if (mSynchronous)
    {
      result = mThreadedTask->result();
      mThreadedTask = nullptr;
      mSynchronousDone = true;
    }

    return result;
  }

  template <size_t N, typename T, typename... Args>
  decltype(auto) invoke(T&, Args&&... args)
  {
    assert(mClient.get());
    using ReturnType =
        typename MessageSetType::template MessageDescriptorAt<N>::ReturnType;
    if (mThreadedTask)
      return ReturnType{Result::Status::kError, "Already processing"};

    mClient->setParams(mHostParams);

    return mClient->template invoke<N>(*mClient.get(),
                                       std::forward<Args>(args)...);
  }


  ProcessState checkProgress(Result& result)
  {
    if (mThreadedTask)
    {
      auto state = mThreadedTask->checkProgress(result);

      if (state == kDone)
      {
        if (!mQueue.empty())
        {
          mThreadedTask = std::unique_ptr<ThreadedTask>(
              new ThreadedTask(mClient, mQueue.front(), false));
          mQueue.pop_front();
          state = kDoneStillProcessing;
          mThreadedTask->mState = kDoneStillProcessing;
        }
        else
        {
          mThreadedTask = nullptr;
        }
      }

      return state;
    }

    return kNoProcess;
  }

  bool synchronous() { return mSynchronous; }
  void setSynchronous(bool synchronous) { mSynchronous = synchronous; }

  void setQueueEnabled(bool queue) { mQueueEnabled = queue; }

  double progress()
  {
    return mThreadedTask ? mThreadedTask->mTask.progress() : 0.0;
  }

  void cancel()
  {
    mQueue.clear();

    if (mThreadedTask) mThreadedTask->cancel(false);
  }

  bool done() const
  {
    return mThreadedTask ? (mThreadedTask->mState == kDone ||
                            mThreadedTask->mState == kDoneStillProcessing)
                         : (mSynchronous && mSynchronousDone);
  }

  void resetDone() { mSynchronousDone = false; }

  ProcessState state() const
  {
    return mThreadedTask ? mThreadedTask->mState : kNoProcess;
  }

  void setCallback(std::function<void()> cb) { mCallback = cb; }

private:
  void swap(NRTThreadingAdaptor&& x, bool includeParams)
  {
    if (mThreadedTask)
    {
      mThreadedTask->cancel(true);
      mThreadedTask.release();
    }

    mThreadedTask.reset(x.mThreadedTask.get());
    using std::swap;
    swap(mQueue, x.mQueue);
    swap(mSynchronous, x.mSynchronous);
    swap(mQueueEnabled, x.mQueueEnabled);
    swap(mCallback, x.mCallback);
    mSynchronousDone = false;
    if (includeParams) mHostParams = std::move(x.mHostParams);
    mClient = std::move(x.mClient);
  }


  struct NRTJob
  {
    ParamSetType          mParams;
    std::function<void()> mCallback;
  };

  struct ThreadedTask
  {

    template <size_t N, typename T>
    struct BufferCopy
    {
      void operator()(typename T::type& param)
      {
        if (param) param = typename T::type(new MemoryBufferAdaptor(param));
      }
    };

    template <size_t N, typename T>
    struct BufferCopyBack
    {
      void operator()(typename T::type& param, Result& r)
      {
        if (param)
          static_cast<MemoryBufferAdaptor*>(param.get())->copyToOrigin(r);
      }
    };

    template <size_t N, typename T>
    struct BufferDelete
    {
      void operator()(typename T::type& param) { param.reset(); }
    };

    ThreadedTask(ClientPointer client, NRTJob& job, bool synchronous)
        : mProcessParams(job.mParams), mState(kNoProcess),
          mClient(client), mContext{mTask}, mCallback{job.mCallback}
    {

      assert(mClient.get() != nullptr); // right?

      std::promise<void> resultPromise;
      mResultReady = resultPromise.get_future();

      mClient->setParams(mProcessParams);
      if (synchronous) { process(std::move(resultPromise)); }
      else
      {
        auto entry = [](ThreadedTask* owner, std::promise<void> result) {
          owner->process(std::move(result));
        };
        mProcessParams.template forEachParamType<BufferT, BufferCopy>();
        mProcessParams.template forEachParamType<InputBufferT, BufferCopy>();
        mState = kProcessing;
        mThread = std::thread(entry, this, std::move(resultPromise));
      }
    }

    Result result() { return mResult; }

    void process(std::promise<void> resultReady)
    {
      assert(mClient.get() != nullptr); // right?
      mState = kProcessing;
      mResult = mClient->template process<float>(mContext);
      resultReady.set_value();
      mState = kDone;
      if (mCallback && !mDetached && !mTask.cancelled()) mCallback();
      if (mDetached) delete this;
    }

    void join() { mThread.join(); }

    void cancel(bool detach)
    {
      mTask.cancel();

      mDetached = detach;

      if (detach && mThread.joinable()) mThread.detach();
    }

    ProcessState checkProgress(Result& result)
    {
      ProcessState state = mState;

      if (state == kDone)
      {
        if (mThread.get_id() != std::thread::id())
        {
          mResultReady.wait();
          result = mResult;
          mThread.join();
        }

        if (!mTask.cancelled())
        {
          if (result.status() != Result::Status::kError)
          {
            Result bufferCopyResult;
            mProcessParams.template forEachParamType<BufferT, BufferCopyBack>(
                bufferCopyResult);
            /// TODO a proper logging system
            if (!bufferCopyResult.ok())
              std::cout << bufferCopyResult.message() << std::endl;
          }
        }
        else
          result = {Result::Status::kCancelled, ""};

        mProcessParams.template forEachParamType<BufferT, BufferDelete>();
        mState = kNoProcess;
      }

      return state;
    }

    ParamSetType          mProcessParams;
    ProcessState          mState;
    std::thread           mThread;
    std::future<void>     mResultReady;
    Result                mResult;
    ClientPointer         mClient;
    FluidTask             mTask;
    FluidContext          mContext;
    bool                  mDetached = false;
    std::function<void()> mCallback;
  };

  FluidContext                  mNRTContext;
  ParamSetType                  mHostParams;
  std::deque<NRTJob>            mQueue;
  bool                          mSynchronous = false;
  bool                          mQueueEnabled = false;
  std::unique_ptr<ThreadedTask> mThreadedTask;
  ClientPointer                 mClient;
  std::function<void()>         mCallback;
  std::atomic<bool>             mSynchronousDone{false};
};

} // namespace client
} // namespace fluid
