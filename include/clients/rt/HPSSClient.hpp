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

#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/HPSS.hpp"
#include "../../algorithms/public/STFT.hpp"
#include <complex>
#include <string>
#include <tuple>

namespace fluid {
namespace client {

namespace hpss {

enum HPSSParamIndex {
  kHSize,
  kPSize,
  kMode,
  kHThresh,
  kPThresh,
  kFFT,
  kMaxFFT,
  kMaxHSize,
  kMaxPSize
};

constexpr auto HPSSParams = defineParameters(
    LongParam("harmFilterSize", "Harmonic Filter Size", 17,
              UpperLimit<kMaxHSize>(), Odd{}, Min(3)),
    LongParam("percFilterSize", "Percussive Filter Size", 31,
              UpperLimit<kMaxPSize>(), Odd{}, Min(3)),
    EnumParam("maskingMode", "Masking Mode", 0, "Classic", "Coupled",
              "Advanced"),
    FloatPairsArrayParam("harmThresh", "Harmonic Filter Thresholds",
                         FrequencyAmpPairConstraint{}),
    FloatPairsArrayParam("percThresh", "Percussive Filter Thresholds",
                         FrequencyAmpPairConstraint{}),
    FFTParam<kMaxFFT>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}),
    LongParam<Fixed<true>>("maxHarmFilterSize", "Maximum Harmonic Filter Size",
                           101, Min(3), Odd{}),
    LongParam<Fixed<true>>("maxPercFilterSize",
                           "Maximum Percussive Filter Size", 101, Min(3),
                           Odd{}));


class HPSSClient : public FluidBaseClient, public AudioIn, public AudioOut
{
public:
  using ParamDescType = decltype(HPSSParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return HPSSParams; }

  HPSSClient(ParamSetViewType& p)
      : mParams{p}, mSTFTBufferedProcess{get<kMaxFFT>(), 1, 3},
        mHPSS{get<kMaxFFT>(), get<kMaxHSize>()}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(3);
    FluidBaseClient::setInputLabels({"audio input"});
    FluidBaseClient::setOutputLabels({"harmonic component", "percussive compoennt", "residual (in modes 1 & 2)"});
  }

  index latency()
  {
    return ((get<kHSize>() - 1) * get<kFFT>().hopSize()) +
           get<kFFT>().winSize();
  }

  void reset()
  {
    mSTFTBufferedProcess.reset();
    mHPSS.init(get<kFFT>().frameSize(), get<kHSize>());
  }


  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {

    if (!input[0].data()) return;

    index nBins = get<kFFT>().frameSize();

    if (!mHPSS.initialized() || mTrackChanges.changed(nBins, get<kHSize>()))
    {
      mHPSS.init(nBins, get<kHSize>());
    }

    mSTFTBufferedProcess.process(
        mParams, input, output, c,
        [&](ComplexMatrixView in, ComplexMatrixView out) {
          mHPSS.processFrame(
              in.row(0), out.transpose(), get<kPSize>(), get<kHSize>(),
              get<kMode>(), get<kHThresh>().value[0].first,
              get<kHThresh>().value[0].second, get<kHThresh>().value[1].first,
              get<kHThresh>().value[1].second, get<kPThresh>().value[0].first,
              get<kPThresh>().value[0].second, get<kPThresh>().value[1].first,
              get<kPThresh>().value[1].second);
        });
  }

private:
  STFTBufferedProcess<ParamSetViewType, kFFT, true> mSTFTBufferedProcess;
  ParameterTrackChanges<index, index>               mTrackChanges;
  algorithm::HPSS                                   mHPSS;
};
} // namespace hpss
using RTHPSSClient = ClientWrapper<hpss::HPSSClient>;

auto constexpr NRTHPSSParams = makeNRTParams<hpss::HPSSClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("harmonic", "Harmonic Buffer"),
    BufferParam("percussive", "Percussive Buffer"),
    BufferParam("residual", "Residual Buffer"));

using NRTHPSSClient =
    NRTStreamAdaptor<hpss::HPSSClient, decltype(NRTHPSSParams), NRTHPSSParams,
                     1, 3>;

using NRTThreadedHPSSClient = NRTThreadingAdaptor<NRTHPSSClient>;

} // namespace client
} // namespace fluid
