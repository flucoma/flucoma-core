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
  kFFT
};

constexpr auto HPSSParams = defineParameters(
    LongParamRuntimeMax<Primary>("harmFilterSize", "Harmonic Filter Size", 17,
               Odd{}, Min(3)),
    LongParamRuntimeMax<Primary>("percFilterSize", "Percussive Filter Size", 31,
               Odd{}, Min(3)),
    EnumParam("maskingMode", "Masking Mode", 0, "Classic", "Coupled",
              "Advanced"),
    FloatPairsArrayParam("harmThresh", "Harmonic Filter Thresholds",
                         FrequencyAmpPairConstraint{}),
    FloatPairsArrayParam("percThresh", "Percussive Filter Thresholds",
                         FrequencyAmpPairConstraint{}),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));


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

  HPSSClient(ParamSetViewType& p, FluidContext const& c)
      : mParams{p}, mSTFTBufferedProcess{get<kFFT>(), 1, 3,
                                         c.hostVectorSize(), c.allocator()},
        mHPSS{get<kFFT>().max(), get<kHSize>().max(), c.allocator()}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(3);
    FluidBaseClient::setInputLabels({"audio input"});
    FluidBaseClient::setOutputLabels({"harmonic component", "percussive component", "residual (in modes 1 & 2)"});
  }

  index latency() const
  {
    return ((get<kHSize>() - 1) * get<kFFT>().hopSize()) +
           get<kFFT>().winSize();
  }

  void reset(FluidContext&)
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
    { mHPSS.init(nBins, get<kHSize>()); }

    mSTFTBufferedProcess.process(
        get<kFFT>(), input, output, c,
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
  STFTBufferedProcess<true>             mSTFTBufferedProcess;
  ParameterTrackChanges<index, index>   mTrackChanges;
  algorithm::HPSS                       mHPSS;
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
