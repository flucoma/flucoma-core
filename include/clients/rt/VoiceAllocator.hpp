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
//#include "../../algorithms/public/RunningStats.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace voiceallocator {

template <typename T>
using HostVector = FluidTensorView<T, 1>;

constexpr auto VoiceAllocatorParams =
    defineParameters(LongParam("history", "History Size", 2, Min(2))); //will be most probably a max num voice and all other params

class VoiceAllocatorClient : public FluidBaseClient, public ControlIn, ControlOut
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

    VoiceAllocatorClient(ParamSetViewType& p, FluidContext&)
      : mParams(p), mInputSize{0}, mSizeTracker{0}
  {
    controlChannelsIn(3);
    controlChannelsOut({2, -1});
    setInputLabels({"input stream", "middle", "third"});
    setOutputLabels({"mean", "sample standard deviation"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

      std::cout << "left: " << input[0] << '\n';
      std::cout << "middle: " << input[1] << '\n';
      std::cout << "right: " << input[2] << '\n';

    bool inputSizeChanged = mInputSize != input[0].size() ;
    bool sizeParamChanged = mSizeTracker.changed(get<0>());

    if(inputSizeChanged|| sizeParamChanged)
    {
      mInputSize = input[0].size();
//      mAlgorithm.init(get<0>(),mInputSize);
    }

//    mAlgorithm.process(input[0],output[0],output[1]);
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
  index mInputSize;
  ParameterTrackChanges<index> mSizeTracker;
};

} // namespace runningstats

using VoiceAllocatorClient = ClientWrapper<voiceallocator::VoiceAllocatorClient>;

} // namespace client
} // namespace fluid
