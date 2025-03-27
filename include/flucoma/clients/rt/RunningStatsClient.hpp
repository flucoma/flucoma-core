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

#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidSource.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/RunningStats.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace runningstats {

template <typename T>
using HostVector = FluidTensorView<T, 1>;

constexpr auto RunningStatsParams =
    defineParameters(LongParam("history", "History Size", 2, Min(2)));

class RunningStatsClient : public FluidBaseClient, public ControlOutFollowsIn
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

  RunningStatsClient(ParamSetViewType& p, FluidContext&)
      : mParams(p), mInputSize{0}, mSizeTracker{0}
  {
    controlChannelsIn(1);
    controlChannelsOut({2, -1});
    setInputLabels({"input stream"});
    setOutputLabels({"mean", "sample standard deviation"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

    bool inputSizeChanged = mInputSize != input[0].size() ;
    bool sizeParamChanged = mSizeTracker.changed(get<0>());

    if(inputSizeChanged|| sizeParamChanged)
    {
      mInputSize = input[0].size();
      mAlgorithm.init(get<0>(),mInputSize);
    }

    mAlgorithm.process(input[0],output[0],output[1]);
  }

  MessageResult<void> clear()
  {     
    mAlgorithm.init(get<0>(),mInputSize);
    return {};
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(makeMessage("clear", &RunningStatsClient::clear));
  }

  index latency() const { return 0; }

private:
  algorithm::RunningStats mAlgorithm;
  index mInputSize;
  ParameterTrackChanges<index> mSizeTracker;
};

} // namespace runningstats

using RunningStatsClient = ClientWrapper<runningstats::RunningStatsClient>;

} // namespace client
} // namespace fluid
