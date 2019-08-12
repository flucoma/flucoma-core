#pragma once

#include "BufferedProcess.hpp"
#include "../../algorithms/public/Loudness.hpp"
#include "../../data/TensorTypes.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"

#include <tuple>

namespace fluid {
namespace client {

using algorithm::Loudness;

class LoudnessClient: public FluidBaseClient, public AudioIn, public ControlOut {

  enum LoudnessParamIndex {
    kKWeighting,
    kTruePeak,
    kWindowSize,
    kHopSize,
    kMaxWindowSize
  };

public:

  FLUID_DECLARE_PARAMS(
    EnumParam("kWeighting", "Apply K-Weighting", 1, "Off","On"),
    EnumParam("truePeak", "Compute True Peak", 1, "Off","On"),
    LongParam("windowSize", "Window Size", 1024, UpperLimit<kMaxWindowSize>()),
    LongParam("hopSize", "Hop Size", 512, Min(1)),
    LongParam<Fixed<true>>("maxWindowSize", "Max Window Size",
              16384, Min(4), PowerOfTwo{})
  ); // 17640 next power of two


  LoudnessClient(ParamSetViewType &p) : mParams(p),
        mAlgorithm{static_cast<int>(get<kMaxWindowSize>())} {
    audioChannelsIn(1);
    controlChannelsOut(2);
    mDescriptors = FluidTensor<double, 1>(2);
  }

  template <typename T>
  void process(std::vector<HostVector<T>> &input, std::vector<HostVector<T>> &output, FluidContext& c,
               bool reset = false) {
    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");
    size_t hostVecSize = input[0].size();
    if (mBufferParamsTracker.changed(hostVecSize, get<kWindowSize>(),
                                     get<kHopSize>())) {
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
        get<kWindowSize>(), get<kHopSize>(), c, reset, [&](RealMatrixView frame) {
          mAlgorithm.processFrame(frame.row(0), mDescriptors,
                                  get<kKWeighting>() == 1,
                                  get<kTruePeak>() == 1);
        });
    output[0](0) = mDescriptors(0);
    output[1](0) = mDescriptors(1);
  }

  size_t latency() { return get<kWindowSize>(); }

  size_t controlRate() { return get<kHopSize>(); }

private:
  Loudness mAlgorithm;
  ParameterTrackChanges<size_t, size_t, size_t> mBufferParamsTracker;
  BufferedProcess mBufferedProcess;
  FluidTensor<double, 1> mDescriptors;
};


using RTLoudnessClient = ClientWrapper<LoudnessClient>;

auto constexpr NRTLoudnessParams =
    makeNRTParams<RTLoudnessClient>({InputBufferParam("source", "Source Buffer")},
                                  {BufferParam("features", "Features Buffer")});

using NRTLoudnessClient =
    NRTControlAdaptor<RTLoudnessClient, decltype(NRTLoudnessParams),
                      NRTLoudnessParams, 1, 1>;

using NRTThreadedLoudnessClient = NRTThreadingAdaptor<NRTLoudnessClient>;

} // namespace client
} // namespace fluid
