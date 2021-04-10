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
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/Loudness.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace loudness {

enum LoudnessParamIndex {
  kKWeighting,
  kTruePeak,
  kWindowSize,
  kHopSize,
  kMaxWindowSize
};

constexpr auto LoudnessParams = defineParameters(
    EnumParam("kWeighting", "Apply K-Weighting", 1, "Off", "On"),
    EnumParam("truePeak", "Compute True Peak", 1, "Off", "On"),
    LongParam("windowSize", "Window Size", 1024, UpperLimit<kMaxWindowSize>()),
    LongParam("hopSize", "Hop Size", 512, Min(1)),
    LongParam<Fixed<true>>("maxWindowSize", "Max Window Size", 16384, Min(4),
                           PowerOfTwo{}) // 17640 next power of two
);

class LoudnessClient : public FluidBaseClient, public AudioIn, public ControlOut
{

public:
  using ParamDescType = decltype(LoudnessParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return LoudnessParams; }

  LoudnessClient(ParamSetViewType& p)
      : mParams(p), mAlgorithm{get<kMaxWindowSize>()}
  {
    audioChannelsIn(1);
    controlChannelsOut(2);
    mDescriptors = FluidTensor<double, 1>(2);
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data() || !output[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(FluidBaseClient::controlChannelsOut()) &&
           "Too few output channels");
    index hostVecSize = input[0].size();
    if (mBufferParamsTracker.changed(hostVecSize, get<kWindowSize>(),
                                     get<kHopSize>(), sampleRate()))
    {
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(get<kWindowSize>(), get<kWindowSize>(),
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::controlChannelsOut());
      mAlgorithm.init(get<kWindowSize>(), sampleRate());
    }
    RealMatrix in(1, hostVecSize);
    in.row(0) = input[0];
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.processInput(
        get<kWindowSize>(), get<kHopSize>(), c, [&](RealMatrixView frame) {
          mAlgorithm.processFrame(frame.row(0), mDescriptors,
                                  get<kKWeighting>() == 1,
                                  get<kTruePeak>() == 1);
        });
    output[0](0) = static_cast<T>(mDescriptors(0));
    output[1](0) = static_cast<T>(mDescriptors(1));
  }

  index latency() { return get<kWindowSize>(); }

  void reset()
  {
    mBufferedProcess.reset();
    mAlgorithm.init(get<kWindowSize>(), sampleRate());
  }

  index controlRate() { return get<kHopSize>(); }

private:
  ParameterTrackChanges<index, index, index, double> mBufferParamsTracker;
  algorithm::Loudness                                mAlgorithm;
  BufferedProcess                                    mBufferedProcess;
  FluidTensor<double, 1>                             mDescriptors;
};
} // namespace loudness

using RTLoudnessClient = ClientWrapper<loudness::LoudnessClient>;

auto constexpr NRTLoudnessParams = makeNRTParams<loudness::LoudnessClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Features Buffer"));

using NRTLoudnessClient =
    NRTControlAdaptor<loudness::LoudnessClient, decltype(NRTLoudnessParams),
                      NRTLoudnessParams, 1, 1>;

using NRTThreadedLoudnessClient = NRTThreadingAdaptor<NRTLoudnessClient>;

} // namespace client
} // namespace fluid
