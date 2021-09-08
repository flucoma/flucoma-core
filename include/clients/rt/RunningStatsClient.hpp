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

#include "../../algorithms/public/IncrementalStats.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/AudioClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace runningstats {

template <typename T>
using HostVector = FluidTensorView<T, 1>;

//enum PitchParamIndex {
//  kAlgorithm,
//  kMinFreq,
//  kMaxFreq,
//  kUnit,
//  kFFT,
//  kMaxFFTSize
//};

constexpr auto RunningStatsParams = defineParameters(
  LongParam("size","History Size",2,Min(1))
);

class RunningStatsClient : public FluidBaseClient,  public ControlIn,  ControlOut
{
public:
  using ParamDescType = decltype(RunningStatsParams);

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
    return RunningStatsParams; 
  }

  RunningStatsClient(ParamSetViewType& p): mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut(2);
    setInputLabels({"list input"});
    setOutputLabels({"stststst"});
  } 
  
  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    mAlgorithm.process(input[0],output[0],output[1]); 
  }
  
  MessageResult<void> clear() { mAlgorithm.clear(); return {}; }
  
  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("clear", &RunningStatsClient::clear)
    );
  }
  
  index latency() { return 0; }
private:
  algorithm::IncrementalStats mAlgorithm; 
};

} // namespace runningstats

using RunningStatsClient = ClientWrapper<runningstats::RunningStatsClient>;

//auto constexpr NRTPitchParams = makeNRTParams<runningstats::RunningStatsClient>(
//    InputBufferParam("source", "Source Buffer"),
//    BufferParam("features", "Features Buffer"));
//
//using NRTPitchClient =
//    NRTControlAdaptor<pitch::PitchClient, decltype(NRTPitchParams),
//                      NRTPitchParams, 1, 1>;
//
//using NRTThreadedPitchClient = NRTThreadingAdaptor<NRTPitchClient>;

} // namespace client
} // namespace fluid
