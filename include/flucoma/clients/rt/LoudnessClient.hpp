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
  kSelect,
  kKWeighting,
  kTruePeak,
  kWindowSize,
  kHopSize,
  kMaxWindowSize
};

constexpr auto LoudnessParams = defineParameters(
    ChoicesParam("select","Selection of Outputs","loudness","peak"),
    EnumParam("kWeighting", "Apply K-Weighting", 1, "Off", "On"),
    EnumParam("truePeak", "Compute True Peak", 1, "Off", "On"),
    LongParam("windowSize", "Window Size", 1024, UpperLimit<kMaxWindowSize>()),
    LongParam("hopSize", "Hop Size", 512, Min(1)),
    LongParam<Fixed<true>>("maxWindowSize", "Max Window Size", 16384, Min(4),
                           PowerOfTwo{}, Max(32768)) // 17640 next power of two
);

class LoudnessClient : public FluidBaseClient, public AudioIn, public ControlOut
{
  static constexpr index mMaxFeatures = 2; 
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

  LoudnessClient(ParamSetViewType& p, FluidContext& c)
      : mParams(p), mAlgorithm{get<kMaxWindowSize>(), c.allocator()},
      mBufferedProcess{get<kMaxWindowSize>(), 0, 1, 0, c.hostVectorSize(), c.allocator()}
  {
    audioChannelsIn(1);
    controlChannelsOut({1,mMaxFeatures});
    setInputLabels({"audio input"});
    setOutputLabels({"loudness and peak amplitude"});
    mDescriptors = FluidTensor<double, 1>(mMaxFeatures);
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data() || !output[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut().size && "No control channels");
    assert(output[0].size() >= FluidBaseClient::controlChannelsOut().size &&
           "Too few output channels");
    assert(input[0].size() == c.hostVectorSize());
    index hostVecSize = input[0].size();
    if (mBufferParamsTracker.changed(hostVecSize, get<kWindowSize>(),
                                     get<kHopSize>(), sampleRate()))
    {
//      mBufferedProcess.hostSize(hostVecSize);
//      mBufferedProcess.maxSize(get<kWindowSize>(), get<kWindowSize>(),
//                               FluidBaseClient::audioChannelsIn(),
//                               FluidBaseClient::controlChannelsOut().size);
      mAlgorithm.init(get<kWindowSize>(), sampleRate(), c.allocator());
      mBufferedProcess = BufferedProcess{get<kMaxWindowSize>(), 0, 1, 0, c.hostVectorSize(), c.allocator()};
    }
    
    RealMatrix in(1, hostVecSize, c.allocator());
    in.row(0) <<= input[0];
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.processInput(
        get<kWindowSize>(), get<kHopSize>(), c, [&](RealMatrixView frame) {
          mAlgorithm.processFrame(frame.row(0), mDescriptors,
                                  get<kKWeighting>() == 1,
                                  get<kTruePeak>() == 1, c.allocator());
        });
    
    auto selection = get<kSelect>();
    index numSelected = asSigned(selection.count());
    index numOuts = std::min<index>(mMaxFeatures,numSelected);
    index i = 0;
    controlChannelsOut({1,numOuts, mMaxFeatures});

    if (selection[0]) output[0](i++) = static_cast<T>(mDescriptors(0));
    if (selection[1]) output[0](i) = static_cast<T>(mDescriptors(1));

    output[0](Slice(numOuts, mMaxFeatures - numOuts)).fill(0);
  }

  index latency() const { return get<kWindowSize>(); }

  void reset(FluidContext& c)
  {
    mBufferedProcess.reset();
    mAlgorithm.init(get<kWindowSize>(), sampleRate(), c.allocator());
  }

  AnalysisSize analysisSettings()
  {
    return { get<kWindowSize>(), get<kHopSize>() }; 
  }


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
