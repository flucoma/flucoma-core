/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../../algorithms/public/SineExtraction.hpp"
#include "../common/AudioClient.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"

#include <tuple>

namespace fluid {
namespace client {

enum SinesParamIndex
{
  kBandwidth,
  kThreshold,
  kMinTrackLen,
  kMagWeight,
  kFreqWeight,
  kFFT,
  kMaxFFTSize
};

extern auto constexpr SinesParams = defineParameters(
    LongParam("bandwidth", "Bandwidth", 76, Min(1),
              FrameSizeUpperLimit<kFFT>()),
    FloatParam("threshold", "Threshold", 0.7, Min(0.0), Max(1.0)),
    LongParam("minTrackLen", "Min Track Length", 15, Min(0)),
    FloatParam("magWeight", "Magnitude Weighting", 0.01, Min(0.0), Max(1.0)),
    FloatParam("freqWeight", "Frequency Weighting", 0.5, Min(0.0), Max(1.0)),
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
        mTrackValues.changed(get<kFFT>().winSize(), get<kFFT>().hopSize(),
                             get<kFFT>().fftSize(), get<kBandwidth>(),
                             get<kMinTrackLen>()))
    {
      // mSinesExtractor.reset(new
      // algorithm::SineExtraction(get<kFFT>().winSize(), get<kFFT>().fftSize(),
      // get<kFFT>().hopSize(),
      //                                                      get<kBandwidth>(),
      //                                                      get<kThreshold>(),
      //                                                      get<kMinTrackLen>(),
      //                                                      get<kMagWeight>(),
      //                                                      get<kFreqWeight>()));
      mSinesExtractor.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                           get<kFFT>().hopSize(), get<kBandwidth>(),
                           get<kThreshold>(), get<kMinTrackLen>(),
                           get<kMagWeight>(), get<kFreqWeight>());
    } else
    {
      mSinesExtractor.setThreshold(get<kThreshold>());
      mSinesExtractor.setMagWeight(get<kMagWeight>());
      mSinesExtractor.setFreqWeight(get<kFreqWeight>());
      mSinesExtractor.setMinTrackLength(get<kMinTrackLen>());
    }

    mSTFTBufferedProcess.process(
        mParams, input, output, c, reset,
        [this](ComplexMatrixView in, ComplexMatrixView out) {
          mSinesExtractor.processFrame(in.row(0), out.transpose());
        });
  }

  size_t latency()
  {
    return get<kFFT>().winSize() +
           (get<kFFT>().hopSize() * get<kMinTrackLen>());
  }

private:
  STFTBufferedProcess<ParamSetViewType, T, kFFT> mSTFTBufferedProcess;
  // std::unique_ptr<algorithm::SineExtraction> mSinesExtractor;
  algorithm::SineExtraction mSinesExtractor{get<kMaxFFTSize>()};
  ParameterTrackChanges<size_t, size_t, size_t, size_t, size_t> mTrackValues;

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
