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

#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidSource.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
// #include "../../algorithms/public/RunningStats.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace voiceallocator {

template <typename T>
using HostVector = FluidTensorView<T, 1>;

enum VoiceAllocatorParamIndex {
  kNVoices,
  kBirthLowThreshold,
  kBirthHighTreshold,
  kMinTrackLen,
  kTrackMethod,
  kTrackMagRange,
  kTrackFreqRange,
  kTrackProb
};

constexpr auto VoiceAllocatorParams = defineParameters(
    LongParamRuntimeMax<Primary>( "numVoices", "Number of Voices", 1, Min(1)),
    FloatParam("birthLowThreshold", "Track Birth Low Frequency Threshold", -24, Min(-144), Max(0)),
    FloatParam("birthHighThreshold", "Track Birth High Frequency Threshold", -60, Min(-144), Max(0)),
    LongParam("minTrackLen", "Minimum Track Length", 1, Min(1)),
    EnumParam("trackMethod", "Tracking Method", 0, "Greedy", "Hungarian"),
    FloatParam("trackMagRange", "Tracking Magnitude Range (dB)", 15., Min(1.), Max(200.)),
    FloatParam("trackFreqRange", "Tracking Frequency Range (Hz)", 50., Min(1.), Max(10000.)),
    FloatParam("trackProb", "Tracking Matching Probability", 0.5, Min(0.0), Max(1.0))
    );

class VoiceAllocatorClient : public FluidBaseClient,
                             public ControlIn,
                             ControlOut
{
    template <typename T>
    using vector = rt::vector<T>;

public:
  using ParamDescType = decltype(VoiceAllocatorParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return VoiceAllocatorParams;
  }

  VoiceAllocatorClient(ParamSetViewType& p, FluidContext& c)
      : mParams(p), mInputSize{0}, mSizeTracker{0},
    mOut0(get<kNVoices>().max(), c.allocator()),
    mOut1(get<kNVoices>().max(), c.allocator()),
    mOut2(get<kNVoices>().max(), c.allocator())
  {
    controlChannelsIn(2);
    controlChannelsOut({3, get<kNVoices>(), get<kNVoices>().max()});
    setInputLabels({"frequencies", "magnitudes"});
    setOutputLabels({"frequencies", "magnitudes", "voice IDs"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    index nVoices = get<kNVoices>();

    bool inputSizeChanged = mInputSize != input[0].size();
    bool sizeParamChanged = mSizeTracker.changed(nVoices);

    if (inputSizeChanged || sizeParamChanged)
    {
      mInputSize = input[0].size();
      controlChannelsOut({3, nVoices}); //update the dynamic out size
        // other initialisation
    }

    // copy in to fixed output array
      mOut2(Slice(0,input[0].size())) <<= input[0];// dummy code so I don't check that the input is smaller than the output - the voice alloc algo should deal with input more cleverly than just copy
      mOut1(Slice(0,input[1].size())) <<= input[1];
      mOut0(Slice(0,input[0].size())) <<= input[0];
      
    //    mAlgorithm.process(input[0],output[0],output[1]);
    output[2](Slice(0,nVoices)) <<= mOut2(Slice(0,nVoices));
    output[1](Slice(0,nVoices)) <<= mOut1(Slice(0,nVoices));
    output[0](Slice(0,nVoices)) <<= mOut0(Slice(0,nVoices));
  }

  MessageResult<void> clear()
  {
    //    mAlgorithm.init(get<0>(),mInputSize);
    return {};
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(makeMessage("clear", &VoiceAllocatorClient::clear));
  }

  index latency() const { return 0; }

private:
  //  algorithm::RunningStats mAlgorithm;
    index                                       mInputSize;
    ParameterTrackChanges<index>                mSizeTracker;
    FluidTensor<double, 1>                      mOut0;
    FluidTensor<double, 1>                      mOut1;
    FluidTensor<double, 1>                      mOut2;
};

} // namespace voiceallocator

using VoiceAllocatorClient =
    ClientWrapper<voiceallocator::VoiceAllocatorClient>;

} // namespace client
} // namespace fluid
