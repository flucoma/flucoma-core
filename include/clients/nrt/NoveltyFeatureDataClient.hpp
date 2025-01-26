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

#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/NoveltyFeature.hpp"

namespace fluid {
namespace client {

enum NoveltyDataParamIndex {kSource, kOffset, kNumFrames, kStartChan, kNumChans, kOutput, kKernelSize,kFilterSize};

auto constexpr NoveltyDataParams = defineParameters(
  InputBufferParam("source","Source Buffer"),
  LongParam("startFrame","Source Offset",0,Min(0)),
  LongParam("numFrames","Number of Frames",-1),
  LongParam("startChan","Start Channel",0,Min(0)),
  LongParam("numChans","Number of Channels",-1),
  BufferParam("output", "Output Buffer"),
  LongParam("kernelSize", "Kernel Size", 3, Min(3), Odd()),
  LongParam("filterSize", "Smoothing Filter Size", 1, Min(1))
 );


class NoveltyFeatureDataClient: public FluidBaseClient, public OfflineIn, public OfflineOut
{

public:
    
  using ParamDescType = decltype(NoveltyDataParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;

  NoveltyFeatureDataClient(ParamSetViewType& p)
  : mParams{p}
  {}

  template <std::size_t N>
  auto& get() noexcept
  {
    return mParams.get().template get<N>();
  }
    
  void setParams(ParamSetViewType& p)
  {
    mParams = p;
  }
    
  static constexpr auto& getParameterDescriptors() { return NoveltyDataParams; }

  index latency()
  {
    index filterSize = get<kFilterSize>();
    if (filterSize % 2) filterSize++;
    return ((get<kKernelSize>() + 1) >> 1) + (filterSize >> 1);
  }
    
  template<typename T>
  Result process(FluidContext&)
  {
    if(!get<kSource>().get())
      return {Result::Status::kError, "No input buffer supplied"};

    BufferAdaptor::ReadAccess source(get<kSource>().get());

    if(!source.exists())
        return {Result::Status::kError, "Input buffer not found"};

    if(!source.valid())
        return {Result::Status::kError, "Can't access input buffer"};

    {
        BufferAdaptor::Access idx(get<kOutput>().get());

        if(!idx.exists())
            return {Result::Status::kError, "Output buffer not found"};
    }

    index numFrames = get<kNumFrames>() == -1
      ? (source.numFrames() - get<kOffset>())
      : get<kNumFrames>();
    index numChannels = get<kNumChans>() == -1
      ? (source.numChans() - get<kStartChan>())
      : get<kNumChans>();
      
    index kernelSize = get<kKernelSize>();
    index filterSize = get<kFilterSize>();
      
    algorithm::NoveltyFeature processor(kernelSize, filterSize);

    auto inputData = FluidTensor<double,1>(numChannels);
    auto outputData = FluidTensor<double,1>(numFrames);
      
    processor.init(kernelSize, filterSize, numChannels);
          
    index preRoll = latency();
    index chanOffset = get<kStartChan>();
    index offset = get<kOffset>();
      
    auto getInput = [&](index idx)
    {
        // FIXME: make this nicer

        for (index j = 0; j < numChannels; j++)
            inputData(j) = source.samps(idx, 1, chanOffset + j)(0);
    };
      
    for (index i = 0; i < preRoll; i++)
    {
        getInput(offset + i);
        processor.processFrame(inputData);
    }
      
    offset += preRoll;
      
    for (index i = 0; i < numFrames - preRoll; i++)
    {
        getInput(offset + i);
        outputData(i) = processor.processFrame(inputData);
    }

    // FIXME: consider if there should be a "default input"
      
    inputData.fill(0.0);

    for (index i = numFrames - preRoll; i < numFrames; i++)
        outputData(i) = processor.processFrame(inputData);
    
    BufferAdaptor::Access idx(get<kOutput>().get());
        
    Result resizeResult =
    idx.resize(numFrames, 1, source.sampleRate());
    if (!resizeResult.ok()) return resizeResult;
        
    idx.samps(0) <<= outputData;
    
    return {Result::Status::kOk,""};
  }

  std::reference_wrapper<ParamSetViewType> mParams;
};

using NRTThreadedNoveltyFeatureDataClient = NRTThreadingAdaptor<ClientWrapper<NoveltyFeatureDataClient>>;

} // namespace client
} // namespace fluid
