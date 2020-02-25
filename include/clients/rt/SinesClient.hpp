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
#include "../../algorithms/public/SineExtraction.hpp"
#include <tuple>

namespace fluid {
namespace client {

enum SinesParamIndex {
  kBandwidth,
  kDeathThreshold,
  kBirthLowThreshold,
  kBirthHighThreshold,
  kMinTrackLen,
  kTrackingMethod,
  kTrackMagRange,
  kTrackFreqRange,
  kTrackProb,
  kFFT,
  kMaxFFTSize
};

extern auto constexpr SinesParams = defineParameters(
    LongParam("bandwidth", "Bandwidth", 76, Min(1),
              FrameSizeUpperLimit<kFFT>()),
    FloatParam("deathThreshold", "Track death threshold", -96, Min(-144), Max(0)),
    FloatParam("birthLowThreshold", "Track birth low frequency threshold", -24, Min(-144), Max(0)),
    FloatParam("birthHighThreshold", "Track birth jigh frequency threshold", -60, Min(-144), Max(0)),
    LongParam("minTrackLen", "Min Track Length", 15, Min(1)),
    EnumParam("trackingMethod", "Tracking method", 0, "Greedy", "Munkres"),
    FloatParam("trackMagRange", "Tracking Magnitude Range (dB)", 15., Min(1.), Max(200.)),
    FloatParam("trackFreqRange", "Tracking Frequency Range (Hz)", 50., Min(1.), Max(10000.)),
    FloatParam("trackProb", "Tracking matching probability", 1.0, Min(0.0), Max(1.0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1,
                          FrameSizeLowerLimit<kBandwidth>()),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));


template <typename T>
class SinesClient : public FluidBaseClient<decltype(SinesParams), SinesParams>,
                    public AudioIn,
                    public AudioOut
{
  using HostVector = FluidTensorView<T, 1>;

public:
  SinesClient(ParamSetViewType& p)
      : FluidBaseClient(p), mSTFTBufferedProcess{get<kMaxFFTSize>(), 1, 2}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(2);
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c, bool reset = false)
  {

    if (!input[0].data()) return;
    if (!output[0].data() && !output[1].data()) return;
    if (!mSinesExtractor.initialized() ||
        mTrackValues.changed(get<kFFT>().winSize(),
                             get<kFFT>().fftSize(), get<kBandwidth>()))
    {
      mSinesExtractor.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                           get<kBandwidth>());
    }
    mSinesExtractor.setDeathThreshold(get<kDeathThreshold>());
    mSinesExtractor.setBirthHighThreshold(get<kBirthHighThreshold>());
    mSinesExtractor.setBirthLowThreshold(get<kBirthLowThreshold>());
    mSinesExtractor.setMinTrackLength(get<kMinTrackLen>());
    mSinesExtractor.setMethod(get<kTrackingMethod>());
    mSinesExtractor.setZetaA(get<kTrackMagRange>());
    mSinesExtractor.setZetaF(get<kTrackFreqRange>());
    mSinesExtractor.setDelta(get<kTrackProb>());

    mSTFTBufferedProcess.process(
        mParams, input, output, c, reset,
        [this](ComplexMatrixView in, ComplexMatrixView out) {
          mSinesExtractor.processFrame(in.row(0), out.transpose(),sampleRate());
        });
  }

  size_t latency()
  {
    return get<kFFT>().winSize() +
           (get<kFFT>().hopSize() * get<kMinTrackLen>());
  }

private:
  STFTBufferedProcess<ParamSetViewType, T, kFFT> mSTFTBufferedProcess;
  algorithm::SineExtraction mSinesExtractor{get<kMaxFFTSize>()};
  ParameterTrackChanges<size_t, size_t, size_t> mTrackValues;

  size_t mWinSize{0};
  size_t mHopSize{0};
  size_t mFFTSize{0};
  size_t mBandwidth{0};
  size_t mMinTrackLen{0};
};

auto constexpr NRTSineParams =
    makeNRTParams<SinesClient>({InputBufferParam("source", "Source Buffer")},
                               {BufferParam("sines", "Sines Buffer"),
                                BufferParam("residual", "Residual Buffer")});

template <typename T>
using NRTSinesClient = NRTStreamAdaptor<SinesClient<T>, decltype(NRTSineParams),
                                        NRTSineParams, 1, 2>;

template <typename T>
using NRTThreadedSinesClient = NRTThreadingAdaptor<NRTSinesClient<T>>;

} // namespace client
} // namespace fluid
