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
class NRTClientWrapper: public FluidBaseClient<Params>, public OfflineIn, public OfflineOut
{

public:
  
  //Host buffers are always float32 (?)
  using HostVector =  FluidTensor<float,1>;
  using HostMatrix =  FluidTensor<float,2>;

  using HostVectorView =  FluidTensorView<float,1>;
  using HostMatrixView =  FluidTensorView<float,2>;
  
  NRTClientWrapper(): FluidBaseClient<Params>(Tuple)
  {
    if(isAudioIn<RTClient>)
      this->audioBuffersIn(mClient.audioChannelsIn());
    if(isAudioOut<RTClient>)
      this->audioBuffersOut(mClient.audioChannelsOut());
  }

  void process(std::vector<BufferProcessSpec>& mInputBuffers, std::vector<BufferProcessSpec>& mOutputBuffers)
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
        return; //error
      
      long requestedFrames= b.nFrames < 0 ? thisInput.numFrames() : b.nFrames;
      if(b.startFrame + requestedFrames > thisInput.numFrames())
        return; //error
      
      long requestedChans= b.nChans < 0 ? thisInput.numChans() : b.nChans;
      if(b.startChan + requestedChans > thisInput.numChans())
        return; //error
  
      inFrames.push_back(b.nFrames < 0 ? thisInput.numFrames() : b.nFrames);
//      startFrames.push_back(b.startFrame);
      inChans.push_back(b.nChans < 0 ? thisInput.numChans() : b.nChans);
//      startChans.push_back(b.startChan);
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
  }
private:
  RTClient  mClient;
};

} //namespace client
} //namespace fluid
