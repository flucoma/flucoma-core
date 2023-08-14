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

constexpr auto VoiceAllocatorParams = defineParameters(LongParam(
    "history", "History Size", 2,
    Min(2))); // will be most probably a max num voice and all other params

class VoiceAllocatorClient : public FluidBaseClient,
                             public ControlIn,
                             ControlOut
{
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
    mOut0(16, c.allocator()),
    mOut1(16, c.allocator()),
    mOut2(16, c.allocator())
  {
    controlChannelsIn(2);
    controlChannelsOut({3, 16, 16});
    setInputLabels({"left", "right"});
    setOutputLabels({"lefto", "middleo", "righto"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

    bool inputSizeChanged = mInputSize != input[0].size();
    bool sizeParamChanged = mSizeTracker.changed(get<0>());

    if (inputSizeChanged || sizeParamChanged)
    {
      mInputSize = input[0].size();
      //      mAlgorithm.init(get<0>(),mInputSize);
    }

    // copy in to fixed output array
      mOut2(Slice(0,input[0].size())) <<= input[0];
      mOut1(Slice(0,input[1].size())) <<= input[1];
      mOut0(Slice(0,input[0].size())) <<= input[0];
      
    //    mAlgorithm.process(input[0],output[0],output[1]);
    output[2] <<= mOut2;
    output[1] <<= mOut1;
    output[0] <<= mOut0;
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

  index latency() { return 0; }

private:
  //  algorithm::RunningStats mAlgorithm;
  index                        mInputSize;
  ParameterTrackChanges<index> mSizeTracker;
    FluidTensor<double, 1>                      mOut0;
    FluidTensor<double, 1>                      mOut1;
    FluidTensor<double, 1>                      mOut2;
};

} // namespace voiceallocator

using VoiceAllocatorClient =
    ClientWrapper<voiceallocator::VoiceAllocatorClient>;

} // namespace client
} // namespace fluid
