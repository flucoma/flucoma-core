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
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/Envelope.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace ampfeature2 {

enum AmpFeature2ParamIndex {
  kFastRampUpTime,
  kFastRampDownTime,
  kSlowRampUpTime,
  kSlowRampDownTime,
  kSilenceThreshold,
  kHiPassFreq,
};

constexpr auto AmpFeature2Params = defineParameters(
    LongParam("fastRampUp", "Fast Envelope Ramp Up Length", 1, Min(1)),
    LongParam("fastRampDown", "Fast Envelope Ramp Down Length", 1, Min(1)),
    LongParam("slowRampUp", "Slow Envelope Ramp Up Length", 100, Min(1)),
    LongParam("slowRampDown", "Slow Envelope Ramp Down Length", 100, Min(1)),
    FloatParam("floor", "Floor value (dB)", -145, Min(-144), Max(144)),
    FloatParam("highPassFreq", "High-Pass Filter Cutoff", 85, Min(0)));

class AmpFeature2Client : public FluidBaseClient, public AudioIn, public AudioOut
{

public:
  using ParamDescType = decltype(AmpFeature2Params);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return AmpFeature2Params; }

  AmpFeature2Client(ParamSetViewType& p) : mParams(p)
  {
    audioChannelsIn(1);
    audioChannelsOut(2);
    FluidBaseClient::setInputLabels({"audio input"});
    FluidBaseClient::setOutputLabels({"fast curve", "slow curve"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

    if (!input[0].data() || (!output[0].data() && !output[1].data())) return;

    double hiPassFreq = std::min(get<kHiPassFreq>() / sampleRate(), 0.5);

    if (!mAlgorithm.initialized())
    {
      mAlgorithm.init(get<kSilenceThreshold>(), hiPassFreq);
    }

    for (index i = 0; i < input[0].size(); i++)
    {
      double fast, slow;
      std::tie(fast, slow) = mAlgorithm.processSampleSeparate(
          input[0](i), get<kSilenceThreshold>(), get<kFastRampUpTime>(),
          get<kSlowRampUpTime>(), get<kFastRampDownTime>(),
          get<kSlowRampDownTime>(), hiPassFreq);
      output[0](i) = static_cast<T>(fast);
      output[1](i) = static_cast<T>(slow);
    }
  }
  index latency() { return 0; }

  void reset()
  {
    double hiPassFreq = std::min(get<kHiPassFreq>() / sampleRate(), 0.5);
    mAlgorithm.init(get<kSilenceThreshold>(), hiPassFreq);
  }

private:
  algorithm::Envelope mAlgorithm;
};

template <typename HostMatrix, typename HostVectorView>
struct NRTAmpFeature2
{
  template <typename Client, typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,
                        OutputList& outputBuffers, index nFrames, index nChans,
                        std::pair<index, index> userPadding, FluidContext& c)
  {
    // expecting a single buffer in and a single buffer out
    assert(inputBuffers.size() == 1);
    assert(outputBuffers.size() == 1);
    HostMatrix inputData(nChans, nFrames);
    HostMatrix outputData(2 * nChans, nFrames);

    // ignoring userPadding as AmpFeature2 has no padding options and no latency

    double sampleRate{0};

    for (index i = 0; i < nChans; ++i)
    {
      BufferAdaptor::ReadAccess thisInput(inputBuffers[0].buffer);
      sampleRate = thisInput.sampleRate();
      inputData.row(i)(Slice(0, nFrames)) <<=
            thisInput.samps(inputBuffers[0].startFrame, nFrames,
                            inputBuffers[0].startChan + i);
      std::vector<HostVectorView> input{inputData.row(i)};

      std::vector<HostVectorView> outputs {
        outputData.row(i*2),
        outputData.row(i*2 + 1),
      };

      client.reset();
      client.process(input, outputs, c);
    }

    BufferAdaptor::Access thisOutput(outputBuffers[0]);
    Result                r = thisOutput.resize(nFrames, nChans * 2, sampleRate);
    if (!r.ok()) return r;
    for (index j = 0; j < nChans * 2; j+=2) {
      thisOutput.samps(j) <<=
        outputData.row(j)(Slice(0, nFrames));
      thisOutput.samps(j + 1) <<=
        outputData.row(j + 1)(Slice(0, nFrames));
    }

    return {};
  }
};
} // namespace ampfeature2

using RTAmpFeature2Client = ClientWrapper<ampfeature2::AmpFeature2Client>;

auto constexpr NRTAmpFeature2Params = makeNRTParams<ampfeature2::AmpFeature2Client>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Feature Buffer"));

using NRTAmpFeature2Client =
    impl::NRTClientWrapper<ampfeature2::NRTAmpFeature2, ampfeature2::AmpFeature2Client, decltype(NRTAmpFeature2Params),
                    NRTAmpFeature2Params, 1, 1>;

using NRTThreadedAmpFeature2Client = NRTThreadingAdaptor<NRTAmpFeature2Client>;

} // namespace client
} // namespace fluid
