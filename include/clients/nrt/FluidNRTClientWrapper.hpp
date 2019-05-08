#pragma once

#include <clients/common/BufferAdaptor.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/SpikesToTimes.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <vector>

namespace fluid{
namespace client{

namespace impl{
//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename B>
auto constexpr makeWrapperInputs(B b)
{
    return defineParameters(std::forward<B>(b),
          LongParam("startFrame", "Source Offset",0, Min(0)),
          LongParam("numFrames","Number of Frames", -1),
          LongParam("startChan","Start Channel",0, Min(0)),
          LongParam("numChans","Number of Channels", -1)
          );
}

template<typename...B>
auto constexpr makeWrapperOutputs(B...b)
{
  return defineParameters(std::forward<B>(b)...);
}

template<typename T,size_t N, size_t...Is>
auto constexpr spitIns(T(&a)[N],std::index_sequence<Is...>)
{
    return makeWrapperInputs(std::forward<T>(a[Is])...);
}

template<typename T,size_t N, size_t...Is>
auto constexpr spitOuts(T(&a)[N],std::index_sequence<Is...>)
{
    return makeWrapperOutputs(std::forward<T>(a[Is])...);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
using BufferSpec = ParamSpec<BufferT,Fixed<false>>;
//////////////////////////////////////////////////////////////////////////////////////////////////////

template<size_t...Is, size_t...Js,typename...Ts, typename...Us>
constexpr auto joinParameterDescriptors(ParameterDescriptorSet<std::index_sequence<Is...>,std::tuple<Ts...>> x,ParameterDescriptorSet<std::index_sequence<Js...>,std::tuple<Us...>> y)
{
  return ParameterDescriptorSet<
            typename JoinOffsetSequence<std::index_sequence<Is...>, std::index_sequence<Js...>>::type,
        std::tuple<Ts...,Us...>> {std::tuple_cat(x.descriptors(),y.descriptors())};
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename HostMatrix,typename HostVectorView>
struct StreamingControl;

template<template <typename, typename> class AdaptorType, class RTClient, typename ParamType, ParamType& PD, size_t Ins, size_t Outs>
class NRTClientWrapper: public OfflineIn, public OfflineOut
{
public:
  //Host buffers are always float32 (?)
  using HostVector     =  FluidTensor<float,1>;
  using HostMatrix     =  FluidTensor<float,2>;
  using HostVectorView =  FluidTensorView<float,1>;
  using HostMatrixView =  FluidTensorView<float,2>;
  static constexpr auto isControl = std::is_same<AdaptorType<HostMatrix,HostVectorView>,StreamingControl<HostMatrix,HostVectorView>>();
  static constexpr auto  decideOuts  =  isControl ? 1: Outs;

  using ParamDescType = ParamType;
  using ParamSetType = ParameterSet<ParamDescType>;
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  using RTParamDescType = typename RTClient::ParamDescType;
  using RTParamSetViewType = ParameterSetView<typename RTClient::ParamDescType>;

  constexpr static ParamDescType& getParameterDescriptors() { return PD; }

  //The client will be accessing its parameter by a bunch of indices that need ofsetting now
//  using Client = RTClient<impl::ParameterSet_Offset<Params,ParamOffset>,T,U>;
// None of that for outputs though

  static constexpr size_t ParamOffset  = (Ins*5) + decideOuts;
  using WrappedClient = RTClient;//<ParameterSet_Offset<Params,ParamOffset>,T>;

  NRTClientWrapper(ParamSetViewType& p)
    : mParams{p}
    , mRealTimeParams{RTClient::getParameterDescriptors(), p.template subset<ParamOffset>()}
    , mClient{mRealTimeParams}
  {}

  template <std::size_t N> auto& get() noexcept { return mParams.template get<N>(); }
//  template <std::size_t N> bool changed() noexcept { return mParams.template changed<N>(); }

  size_t audioChannelsIn()    const noexcept { return 0; }
  size_t audioChannelsOut()   const noexcept { return 0; }
  size_t controlChannelsIn()  const noexcept { return 0; }
  size_t controlChannelsOut() const noexcept { return 0; }
  ///Map delegate audio / control channels to audio buffers
  size_t audioBuffersIn()  const noexcept { return mClient.audioChannelsIn();  }
  size_t audioBuffersOut() const noexcept { return isControl ? 1 : mClient.audioChannelsOut(); }

  Result process()
  {

    
    auto constexpr inputCounter = std::make_index_sequence<Ins>();
    auto constexpr outputCounter = std::make_index_sequence<decideOuts>();

    auto inputBuffers  = fetchInputBuffers(inputCounter);
    auto outputBuffers = fetchOutputBuffers(outputCounter);

    std::array<intptr_t,Ins> inFrames;
    std::array<intptr_t,Ins> inChans;

    //check buffers exist
    int count = 0;
    for(auto&& b: inputBuffers)
    {
      if(!b.buffer)
        return {Result::Status::kError, "Input buffer not set"}; //error
      
      BufferAdaptor::Access thisInput(b.buffer);
      
      if(!thisInput.exists())
        return {Result::Status::kError, "Input buffer ", b.buffer, "."} ; //error

      if(!thisInput.valid())
        return {Result::Status::kError, "Input buffer ", b.buffer, "invalid (possibly zero-size?)"} ; //error

      intptr_t requestedFrames= b.nFrames < 0 ? thisInput.numFrames() : b.nFrames;
      if(b.startFrame + requestedFrames > thisInput.numFrames())
        return {Result::Status::kError, "Input buffer ", b.buffer, ": not enough frames" }; //error

      intptr_t requestedChans= b.nChans < 0 ? thisInput.numChans() : b.nChans;
      if(b.startChan + requestedChans > thisInput.numChans())
        return {Result::Status::kError, "Input buffer ", b.buffer, ": not enough channels" }; //error

      inFrames[count] = b.nFrames < 0 ? thisInput.numFrames() : b.nFrames;
      inChans[count] =  b.nChans < 0 ? thisInput.numChans() : b.nChans ;
      mClient.sampleRate(thisInput.sampleRate());
      count++;
    }
    
    if(std::all_of(outputBuffers.begin(), outputBuffers.end(),[](auto& b){return b == nullptr; }))
      return {Result::Status::kError, "No valid output has been set" }; //error
    

    size_t numFrames   = *std::min_element(inFrames.begin(),inFrames.end());
    size_t numChannels = *std::min_element(inChans.begin(), inChans.end());
    
    AdaptorType<HostMatrix,HostVectorView>::process(mClient,inputBuffers,outputBuffers,numFrames,numChannels);

    return {Result::Status::kOk,""};
  }
private:

  template<size_t I>
  BufferProcessSpec fetchInputBuffer()
  {
    return {get<I>().get(),get<I+1>(), get<I+2>(),get<I+3>(),get<I+4>()};
  }

  template<size_t...Is>
  std::array<BufferProcessSpec, sizeof...(Is)> fetchInputBuffers(std::index_sequence<Is...>)
  {
    return {fetchInputBuffer<Is*5>()...};
  }

  template<size_t...Is>
  std::array<BufferAdaptor*,sizeof...(Is)> fetchOutputBuffers(std::index_sequence<Is...>)
  {
    return {get<Is + (Ins*5)>().get()...};
  }

  RTParamSetViewType    mRealTimeParams;
  ParamSetViewType&     mParams;
  WrappedClient         mClient;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename HostMatrix, typename HostVectorView>
struct Streaming
{
  template <typename Client,typename InputList, typename OutputList>
  static void process(Client& client, InputList& inputBuffers,OutputList& outputBuffers, size_t nFrames, size_t nChans, size_t outputDownsampleFactor = 1)
  {
    //To account for process latency we need to copy the buffers with padding
    std::vector<HostMatrix> outputData;
    std::vector<HostMatrix> inputData;

    outputData.reserve(outputBuffers.size());
    inputData.reserve(inputBuffers.size());

    size_t padding = client.latency();

    std::fill_n(std::back_inserter(outputData), outputBuffers.size(), HostMatrix(nChans,nFrames + padding));
    std::fill_n(std::back_inserter(inputData), inputBuffers.size(), HostMatrix(nChans,nFrames + padding));

    double sampleRate{0};

    for(int i = 0; i < nChans; ++i)
    {
      std::vector<HostVectorView> inputs;
      inputs.reserve(inputBuffers.size());
      for(int j = 0; j < inputBuffers.size(); ++j)
      {
        BufferAdaptor::Access thisInput(inputBuffers[j].buffer);
        if (i==0 && j==0) sampleRate = thisInput.sampleRate();
        inputData[j].row(i)(Slice(0,nFrames)) = thisInput.samps(inputBuffers[j].startFrame,nFrames,inputBuffers[j].startChan + i);
        inputs.emplace_back(inputData[j].row(i));
      }

      std::vector<HostVectorView> outputs;
      outputs.reserve(outputBuffers.size());
      for(int j = 0; j < outputBuffers.size(); ++j)
        outputs.emplace_back(outputData[j].row(i));

      client.process(inputs,outputs);
    }

    for(int i = 0; i < outputBuffers.size(); ++i)
    {
      if(!outputBuffers[i]) continue;
      BufferAdaptor::Access thisOutput(outputBuffers[i]);
      thisOutput.resize(nFrames,nChans,1,sampleRate);
      for(int j = 0; j < nChans; ++j)
        thisOutput.samps(j) = outputData[i].row(j)(Slice(padding));
    }
  }
};
//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename HostMatrix,typename HostVectorView>
struct StreamingControl
{
  template <typename Client,typename InputList, typename OutputList>
  static void process(Client& client, InputList& inputBuffers,OutputList& outputBuffers, size_t nFrames, size_t nChans)
  {
        //To account for process latency we need to copy the buffers with padding
      std::vector<HostMatrix> inputData;
      size_t nFeatures = client.controlChannelsOut();
//      outputData.reserve(nFeatures);
      inputData.reserve(inputBuffers.size());

      size_t padding = client.latency();
      size_t controlRate = client.controlRate();
      size_t nHops = (nFrames + padding) / controlRate;

      //in contrast to the plain streaming case, we're going to call process() iteratively with a vector size = the control vector size, so we get KR where expected
      //TODO make this whole mess less baroque and opaque

      std::fill_n(std::back_inserter(inputData), inputBuffers.size(), HostMatrix(nChans,nFrames+padding));
      HostMatrix outputData(nChans * nFeatures, nHops);
      double sampleRate{0};
      //Copy input data
      for(int i = 0; i < nChans; ++i)
      {
        for(int j = 0; j < inputBuffers.size(); ++j)
        {
          BufferAdaptor::Access thisInput(inputBuffers[j].buffer);
          if(i==0 && j==0) sampleRate = thisInput.sampleRate();
          inputData[j].row(i)(Slice(0,nFrames)) = thisInput.samps(inputBuffers[j].startFrame,nFrames,inputBuffers[j].startChan + i);
        }
      }
    
    for(int i = 0; i < nChans; ++i)
    {
      for(int j = 0; j < nHops; ++j )
      {
        size_t t = j * controlRate;
        std::vector<HostVectorView> inputs;
        inputs.reserve(inputBuffers.size());
        std::vector<HostVectorView> outputs;
        outputs.reserve(outputBuffers.size());
        for(int k = 0; k < inputBuffers.size(); ++k)  inputs.emplace_back(inputData[k].row(i)(Slice(t,controlRate)));
        for(int k = 0; k < nFeatures; ++k) outputs.emplace_back(outputData.row(k + i*nFeatures)(Slice(j,1))); 
        client.process(inputs,outputs);
      }
    }
    
    BufferAdaptor::Access thisOutput(outputBuffers[0]);
    thisOutput.resize(nHops - 1,nChans * nFeatures,1,sampleRate / controlRate);

    for(int i = 0; i < nFeatures; ++i)
    {
      for(int j = 0; j < nChans; ++j)
        thisOutput.samps(i + j * nFeatures) = outputData.row(i + j * nFeatures)(Slice(1));
    }
    
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename HostMatrix,typename HostVectorView>
struct Slicing
{
  template <typename Client,typename InputList, typename OutputList>
  static void process(Client& client, InputList& inputBuffers,OutputList& outputBuffers, size_t nFrames, size_t nChans)
  {

    assert(inputBuffers.size() == 1);
    assert(outputBuffers.size() == 1);
    size_t padding = client.latency();
    HostMatrix monoSource(1,nFrames + padding);

    BufferAdaptor::Access src(inputBuffers[0].buffer);
    // Make a mono sum;
    for (size_t i = 0; i < nChans; ++i)
      monoSource.row(0)(Slice(0,nFrames)).apply(src.samps(i), [](float &x, float y) { x += y; });

    HostMatrix onsetPoints(1,nFrames + padding);

    std::vector<HostVectorView> input  {monoSource.row(0)};
    std::vector<HostVectorView> output {onsetPoints.row(0)};

    client.process(input,output);

    impl::spikesToTimes(onsetPoints(0,Slice(padding,nFrames)).row(0), outputBuffers[0], 1, inputBuffers[0].startFrame, nFrames,src.sampleRate());
  }
};

} //namespace impl
//////////////////////////////////////////////////////////////////////////////////////////////////////

template<class RTClient, typename Params, Params& PD, size_t Ins, size_t Outs>
using NRTStreamAdaptor = impl::NRTClientWrapper<impl::Streaming, RTClient, Params, PD, Ins, Outs>;

template<class RTClient,typename Params, Params& PD, size_t Ins, size_t Outs>
using NRTSliceAdaptor = impl::NRTClientWrapper<impl::Slicing, RTClient, Params, PD, Ins, Outs>;

template<class RTClient,typename Params, Params& PD, size_t Ins, size_t Outs>
using NRTControlAdaptor = impl::NRTClientWrapper<impl::StreamingControl, RTClient, Params, PD, Ins, Outs>;


//////////////////////////////////////////////////////////////////////////////////////////////////////

template<template <typename T> class RTClient, size_t Ms>
auto constexpr makeNRTParams(impl::BufferSpec&& in, impl::BufferSpec(&& out)[Ms])
{
  return impl::joinParameterDescriptors(impl::joinParameterDescriptors(impl::makeWrapperInputs(in), impl::spitOuts(out, std::make_index_sequence<Ms>())), RTClient<double>::getParameterDescriptors());
}

} //namespace client
} //namespace fluid
