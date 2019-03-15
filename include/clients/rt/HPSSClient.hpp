#pragma once

#include <algorithms/public/RTHPSS.hpp>
#include <algorithms/public/STFT.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/rt/BufferedProcess.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <complex>
#include <string>
#include <tuple>

namespace fluid {
namespace client {

enum HPSSParamIndex { kHSize, kPSize, kMode, kHThresh, kPThresh, kFFT, kMaxFFT, kMaxHSize,kMaxPSize };

auto constexpr HPSSParams = defineParameters(
    LongParam("hFiltSize", "Harmonic Filter Size", 17, UpperLimit<kMaxHSize>(),Odd{}, Min(3)),
    LongParam("pFiltSize", "Percussive Filter Size", 31, UpperLimit<kMaxPSize>(),Odd{}, Min(3)),
    EnumParam("modeFlag", "Masking Mode", 0, "Classic", "Coupled", "Advanced"),
    FloatPairsArrayParam("hThresh", "Harmonic Filter Thresholds", FrequencyAmpPairConstraint{}),
    FloatPairsArrayParam("pThresh", "Percussive Filter Thresholds", FrequencyAmpPairConstraint{}),
    FFTParam<kMaxFFT>("fft","FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384) ,
    LongParam<Fixed<true>>("maxHFlitSize", "Maximum Harmonic Filter Size", 101, LowerLimit<kHSize>(), Odd{}),
    LongParam<Fixed<true>>("maxPFiltSize", "Maximum Percussive Filter Size", 101,LowerLimit<kPSize>(), Odd{})
);

template <typename T>
class HPSSClient : public FluidBaseClient<decltype(HPSSParams), HPSSParams>, public AudioIn, public AudioOut
{
  using data_type       = FluidTensorView<T, 2>;
  using complex         = FluidTensorView<std::complex<T>, 1>;
  using HostVector      = HostVector<T>;

public:

  HPSSClient(ParamSetType& p)
    : FluidBaseClient(p), mSTFTBufferedProcess{get<kMaxFFT>(),1,3}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(3);
  }

  size_t latency() { return ((get<kHSize>() - 1) * get<kFFT>().hopSize()) +  get<kFFT>().winSize(); }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {
    if (!input[0].data()) return;

    int nBins = get<kFFT>().frameSize();

    if (mTrackChangesAlgo.changed(nBins, get<kMaxPSize>(), get<kMaxHSize>()))
    {
        mHPSS.init(nBins, get<kMaxPSize>(), get<kMaxHSize>(), get<kPSize>(), get<kHSize>(),
            get<kMode>(), get<kHThresh>()[0].first, get<kHThresh>()[0].second,
            get<kHThresh>()[1].first, get<kHThresh>()[1].second, get<kPThresh>()[0].first,
            get<kPThresh>()[0].second, get<kPThresh>()[1].first, get<kPThresh>()[0].second);
    }
    else
    {
      mHPSS.setVSize(get<kPSize>());
      if(mTrackHSize.changed(get<kHSize>())) mHPSS.setHSize(get<kHSize>());
      
      mHPSS.setHThresholdX1(get<kHThresh>()[0].first);
      mHPSS.setHThresholdY1(get<kHThresh>()[0].second);

      mHPSS.setHThresholdX2(get<kHThresh>()[1].first);
      mHPSS.setHThresholdY2(get<kHThresh>()[1].second);

      mHPSS.setPThresholdX1(get<kPThresh>()[0].first);
      mHPSS.setPThresholdY1(get<kPThresh>()[0].second);

      mHPSS.setPThresholdX2(get<kPThresh>()[1].first);
      mHPSS.setPThresholdY2(get<kPThresh>()[1].second);
      
      mHPSS.setMode(get<kMode>()); 
      
    }

    mSTFTBufferedProcess.process(mParams, input, output,
        [&](ComplexMatrixView in, ComplexMatrixView out)
        {
            mHPSS.processFrame(in.row(0), out.transpose());
        });
  }

private:
  STFTBufferedProcess<ParamSetType, T, kFFT, true> mSTFTBufferedProcess;
  ParameterTrackChanges<size_t, size_t, size_t> mTrackChangesAlgo;
  ParameterTrackChanges<size_t> mTrackHSize;
  algorithm::RTHPSS mHPSS;
};
/*
template <typename Params, typename T, typename U>
using NRTHPSS = NRTStreamAdaptor<HPSSClient,Params,T,U,1,3>;

auto constexpr NRTHPSSParams = impl::makeNRTParams(
    {BufferParam("srcBuf", "Source Buffer")},
    {BufferParam("harmBuf","Harmonic Buffer"),BufferParam("percBuf","Percussive Buffer"),BufferParam("resBuf", "Residual Buffer")},
    HPSSParams);
*/
} // namespace client
} // namespace fluid
