#pragma once

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/BufferAdaptor.hpp>
#include <clients/common/OfflineClient.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <vector>

namespace fluid{
namespace client{

namespace impl{
template<typename RTClient, template <typename,typename> class ProcessType>
class NRTClientWrapper: public OfflineIn, public OfflineOut
{
public:
  
  //Host buffers are always float32 (?)
  using HostVector =  FluidTensor<float,1>;
  using HostMatrix =  FluidTensor<float,2>;

  using HostVectorView =  FluidTensorView<float,1>;
  using HostMatrixView =  FluidTensorView<float,2>;
  
  NRTClientWrapper() = default;

  ///Delegate FluidBaseClient interface back to mClient
  using ValueTuple = typename RTClient::ValueTuple;
  using ParamType = typename RTClient::ParamType;
  using ParamIndexList = typename RTClient::ParamIndexList;
  using UnderlyingTypeTuple = typename RTClient::UnderlyingTypeTuple; 
  
  template <template <size_t N, typename T> class Func>
  static void iterateParameterDescriptors(ParamType params)
  {
    RTClient::template iterateParameterDescriptors<Func>(params);
  }
  
 auto validateParameters(UnderlyingTypeTuple values)
 {
    return mClient.validateParameters(values);
 }
 
  template <template <size_t N, typename T> class Func, typename Values>
  auto setParameterValues(Values *v, bool reportage)
  {
    return mClient.template setParameterValues<Func,Values>(v,reportage);
  }

  template<template <size_t N, typename T> class Func, typename...Args>
  void forEachParam(Args&&...args)
  {
    mClient.template forEachParam<Func>(std::forward<Args>(args)...); 
  }

  template <template <size_t N, typename T> class Func, typename Values>
  auto checkParameterValues(Values* v) { return mClient.template checkParameterValues<Func,Values>(v); }
  
  template <size_t N> auto setter(Result* r) noexcept { return mClient.template setter<N>(r); }
  template <std::size_t N> auto get() noexcept { return mClient.template get<N>(); }
  template <std::size_t N> bool changed() noexcept { return mClient.template changed<N>(); }
  
  size_t audioChannelsIn() const noexcept {return 0;}
  size_t audioChannelsOut() const noexcept {return 0;}
  size_t controlChannelsIn() const noexcept {return 0;}
  size_t controlChannelsOut() const noexcept {return 0;}
  ///Map delegate audio channels to audio buffers
  size_t audioBuffersIn() const noexcept { return mClient.audioChannelsIn();}
  size_t audioBuffersOut() const noexcept { return mClient.audioChannelsOut();}

  Result process(std::vector<BufferProcessSpec>& inputBuffers, std::vector<BufferProcessSpec>& outputBuffers)
  {
  
    std::vector<long> inFrames;
    std::vector<long> inChans;
//    std::vector<long> startFrames;
//    std::vector<long> startChans;
    
    inFrames.reserve(inputBuffers.size());
    inChans.reserve(inputBuffers.size());
    //check buffers exist
    for(auto&& b: inputBuffers)
    {
      BufferAdaptor::Access thisInput(b.buffer);
      if(!thisInput.exists() && !thisInput.valid())
        return {Result::Status::kError, "Input buffer ", b.buffer, " not found or invalid."} ; //error
      
      long requestedFrames= b.nFrames < 0 ? thisInput.numFrames() : b.nFrames;
      if(b.startFrame + requestedFrames > thisInput.numFrames())
        return {Result::Status::kError, "Input buffer ", b.buffer, ": not enough frames" };
      
      long requestedChans= b.nChans < 0 ? thisInput.numChans() : b.nChans;
      if(b.startChan + requestedChans > thisInput.numChans())
        return {Result::Status::kError, "Input buffer ", b.buffer, ": not enough channels" }; //error
  
      inFrames.push_back(b.nFrames < 0 ? thisInput.numFrames() : b.nFrames);
      inChans.push_back(b.nChans < 0 ? thisInput.numChans() : b.nChans);
    }
    
    size_t numFrames = *std::min_element(inFrames.begin(),inFrames.end());
    size_t numChannels = *std::min_element(inChans.begin(), inChans.end());
    
    ProcessType<HostMatrix,HostVectorView>::process(mClient,inputBuffers,outputBuffers,numFrames,numChannels);
    
    return {}; 
  }
private:
  RTClient  mClient;
};

template<typename HostMatrix, typename HostVectorView>
struct Streaming
{
  template <typename Client>
  static void process(Client& client, std::vector<BufferProcessSpec>& inputBuffers,std::vector<BufferProcessSpec>& outputBuffers, size_t nFrames, size_t nChans)
  {
  
    std::vector<HostMatrix> outputData;
    outputData.reserve(outputBuffers.size());
    std::fill_n(std::back_inserter(outputData), outputBuffers.size(), HostMatrix(nChans,nFrames));
    for(int i = 0; i < nChans; ++i)
    {
      std::vector<HostVectorView> inputs;
      inputs.reserve(inputBuffers.size());
      for(int j = 0; j < inputBuffers.size(); ++j)
      {
        BufferAdaptor::Access thisInput(inputBuffers[j].buffer);
        inputs.emplace_back(thisInput.samps(inputBuffers[j].startFrame,nFrames,inputBuffers[j].startChan + i));
      }
      
      std::vector<HostVectorView> outputs;
      outputs.reserve(outputBuffers.size());
      for(int j = 0; j < outputBuffers.size(); ++j)
        outputs.emplace_back(outputData[j].row(i));
      
      client.process(inputs,outputs);
    }

    for(int i = 0; i < outputBuffers.size(); ++i)
    {
      BufferAdaptor::Access thisOutput(outputBuffers[i].buffer);
      thisOutput.resize(nFrames,nChans,1);
      for(int j = 0; j < nChans; ++j)
        thisOutput.samps(j) = outputData[i].row(j);
    }
  }
};

template<typename HostMatrix,typename HostVectorView>
struct Slicing
{
  template <typename Client>
  static void process(Client& client, std::vector<BufferProcessSpec>& inputBuffers,std::vector<BufferProcessSpec>& outputBuffers, size_t nFrames, size_t nChans)
  {
    
    assert(inputBuffers.size() == 1);
    assert(outputBuffers.size() == 1);
    HostMatrix monoSource(1,nFrames);
    BufferAdaptor::Access src(inputBuffers[0].buffer);
    // Make a mono sum;
    for (size_t i = 0; i < nChans; ++i)
      monoSource.apply(src.samps(i), [](float &x, float y) { x += y; });
  
    HostMatrix onsetPoints(1,nFrames);

    std::vector<HostVectorView> input  {monoSource.row(0)};
    std::vector<HostVectorView> output {onsetPoints.row(0)};
    
    client.process(input,output);
    
    size_t numSpikes =
        std::accumulate(onsetPoints.begin(), onsetPoints.end(), 0);

    // Arg sort
    std::vector<size_t> indices(onsetPoints.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&onsetPoints](size_t i1, size_t i2) {
      return onsetPoints[0][i1] > onsetPoints[0][i2];
    });

    // Now put the gathered indicies into ascending order
    std::sort(indices.begin(), indices.begin() + numSpikes);

    // Add model offset
    std::transform(indices.begin(), indices.begin() + numSpikes,
                   indices.begin(),
                   [&](size_t x) -> size_t { return x + inputBuffers[0].startFrame; });

    // insert leading <offset> and num_frames
    indices.insert(indices.begin() + numSpikes, inputBuffers[0].startFrame + nFrames);
    indices.insert(indices.begin(), inputBuffers[0].startFrame);


    BufferAdaptor::Access trans(outputBuffers[0].buffer);
    trans.resize(numSpikes + 2, 1, 1);
    trans.samps(0) =
        FluidTensorView<size_t, 1>{indices.data(), 0, numSpikes + 2};
  }
};

} //namespace impl

template<typename RTClient> using NRTStreamAdaptor = impl::NRTClientWrapper<RTClient, impl::Streaming>;
template<typename RTClient> using NRTSliceAdaptor = impl::NRTClientWrapper<RTClient, impl::Slicing>;


} //namespace client
} //namespace fluid
