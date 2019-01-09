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


template<typename RTClient,  typename Params, Params& Tuple>
class NRTClientWrapper: /*public FluidBaseClient<Params>,*/ public OfflineIn, public OfflineOut
{

public:
  
  //Host buffers are always float32 (?)
  using HostVector =  FluidTensor<float,1>;
  using HostMatrix =  FluidTensor<float,2>;

  using HostVectorView =  FluidTensorView<float,1>;
  using HostMatrixView =  FluidTensorView<float,2>;
  
  NRTClientWrapper()/*: FluidBaseClient<Params>(Tuple)*/
  {
//    if(isAudioIn<RTClient>)
//      this->audioBuffersIn(mClient.audioChannelsIn());
//    if(isAudioOut<RTClient>)
//      this->audioBuffersOut(mClient.audioChannelsOut());
  }

  ///Delegate FluidBaseClient interface back to mClient

  using ValueTuple = typename RTClient::ValueTuple;
  using ParamType = typename RTClient::ParamType;
  using ParamIndexList = typename RTClient::ParamIndexList;
  using UnderlyingTypeTuple = typename RTClient::UnderlyingTypeTuple; 
  
  template <template <size_t N, typename T> class Func>
  static void iterateParameters(ParamType params)
  {
    RTClient::template iterateParameters<Func>(params);
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

  Result process(std::vector<BufferProcessSpec>& mInputBuffers, std::vector<BufferProcessSpec>& mOutputBuffers)
  {
  
    std::vector<long> inFrames;
    std::vector<long> inChans;
//    std::vector<long> startFrames;
//    std::vector<long> startChans;
    
    inFrames.reserve(mInputBuffers.size());
    inChans.reserve(mInputBuffers.size());
    //check buffers exist
    for(auto&& b: mInputBuffers)
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
    
    std::vector<HostMatrix> outputData;
    outputData.reserve(mOutputBuffers.size());
    std::fill_n(std::back_inserter(outputData), mOutputBuffers.size(), HostMatrix(numChannels,numFrames));
    
    for(int i = 0; i < numChannels; ++i)
    {
      std::vector<HostVectorView> inputs;
      inputs.reserve(mInputBuffers.size());
      for(int j = 0; j < mInputBuffers.size(); ++j)
      {
        BufferAdaptor::Access thisInput(mInputBuffers[j].buffer);
        inputs.emplace_back(thisInput.samps(mInputBuffers[j].startFrame,numFrames,mInputBuffers[j].startChan + i));
      }
      
      std::vector<HostVectorView> outputs;
      outputs.reserve(mOutputBuffers.size());
      for(int j = 0; j < mOutputBuffers.size(); ++j)
        outputs.emplace_back(outputData[j].row(i));
      
      mClient.process(inputs,outputs);
    }
    
    for(int i = 0; i < mOutputBuffers.size(); ++i)
    {
      BufferAdaptor::Access thisOutput(mOutputBuffers[i].buffer);
      thisOutput.resize(numFrames,numChannels,1);
      for(int j = 0; j < numChannels; ++j)
        thisOutput.samps(j) = outputData[i].row(j);
    }
    return {}; 
  }
private:
  RTClient  mClient;
};

} //namespace client
} //namespace fluid
