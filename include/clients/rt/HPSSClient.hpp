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

template <typename Params, typename T, typename U = T>
class HPSSClient : public FluidBaseClient<Params>, public AudioIn, public AudioOut
{

  using data_type  = FluidTensorView<T, 2>;
  using complex    = FluidTensorView<std::complex<T>, 1>;
  using HostVector = HostVector<U>;

public:

  HPSSClient(Params& p)
    : mParams{p}, FluidBaseClient<Params>(p), mSTFTBufferedProcess{param<kMaxFFT>(p),1,3}
  {
    FluidBaseClient<Params>::audioChannelsIn(1);
    FluidBaseClient<Params>::audioChannelsOut(3);
  }

  size_t latency() { return ((param<kHSize>(mParams) - 1) * param<kFFT>(mParams).hopSize()) +  param<kFFT>(mParams).winSize(); }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {
    if (!input[0].data()) return;

    int nBins = param<kFFT>(mParams).frameSize();

    if (mTrackChangesAlgo.changed(nBins, param<kMaxPSize>(mParams), param<kMaxHSize>(mParams)))
    {
        mHPSS.init(nBins, param<kMaxPSize>(mParams), param<kMaxHSize>(mParams), param<kPSize>(mParams), param<kHSize>(mParams),
            param<kMode>(mParams), param<kHThresh>(mParams)[0].first, param<kHThresh>(mParams)[0].second,
            param<kHThresh>(mParams)[1].first, param<kHThresh>(mParams)[1].second, param<kPThresh>(mParams)[0].first,
            param<kPThresh>(mParams)[0].second, param<kPThresh>(mParams)[1].first, param<kPThresh>(mParams)[0].second);
    }
    else
    {
      mHPSS.setVSize(param<kPSize>(mParams));
      if(mTrackHSize.changed(param<kHSize>(mParams))) mHPSS.setHSize(param<kHSize>(mParams));
      
      mHPSS.setHThresholdX1(param<kHThresh>(mParams)[0].first);
      mHPSS.setHThresholdY1(param<kHThresh>(mParams)[0].second);

      mHPSS.setHThresholdX2(param<kHThresh>(mParams)[1].first);
      mHPSS.setHThresholdY2(param<kHThresh>(mParams)[1].second);

      mHPSS.setPThresholdX1(param<kPThresh>(mParams)[0].first);
      mHPSS.setPThresholdY1(param<kPThresh>(mParams)[0].second);

      mHPSS.setPThresholdX2(param<kPThresh>(mParams)[1].first);
      mHPSS.setPThresholdY2(param<kPThresh>(mParams)[1].second);
      
      mHPSS.setMode(param<kMode>(mParams)); 
      
    }

    mSTFTBufferedProcess.process(mParams, input, output,
        [&](ComplexMatrixView in, ComplexMatrixView out)
        {
            mHPSS.processFrame(in.row(0), out.transpose());
        });
  }

private:
  Params& mParams;
  STFTBufferedProcess<Params, U, kFFT, true> mSTFTBufferedProcess;
  ParameterTrackChanges<size_t, size_t, size_t> mTrackChangesAlgo;
  ParameterTrackChanges<size_t> mTrackHSize;
  algorithm::RTHPSS mHPSS;
};

template <typename Params, typename T, typename U>
using NRTHPSS = NRTStreamAdaptor<HPSSClient,Params,T,U,1,3>;

auto constexpr NRTHPSSParams = impl::makeNRTParams(
    {BufferParam("srcBuf", "Source Buffer")},
    {BufferParam("harmBuf","Harmonic Buffer"),BufferParam("percBuf","Percussive Buffer"),BufferParam("resBuf", "Residual Buffer")},
    HPSSParams);

} // namespace client
} // namespace fluid
