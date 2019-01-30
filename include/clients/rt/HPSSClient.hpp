#pragma once

#include <algorithms/public/RTHPSS.hpp>
#include <algorithms/public/STFT.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/rt/BufferedProcess.hpp>
#include <complex>
#include <string>
#include <tuple>

namespace fluid {
namespace client {

enum HPSSParamIndex { kHSize, kPSize, kMode, kHThresh, kPThresh, kWinSize, kHopSize, kFFTSize, kMaxWin, kMaxHSize,kMaxPSize };

auto constexpr HPSSParams = std::make_tuple(
    LongParam("hSize", "Harmonic Filter Size", 17, UpperLimit<kMaxHSize>(),Odd{}, Min(3)),
    LongParam("pSize", "Percussive Filter Size", 31, UpperLimit<kMaxPSize>(),Odd{}, Min(3)),
    EnumParam("modeFlag", "Masking Mode", 0, "Classic", "Coupled", "Advanced"),
    FloatPairsArrayParam("hThresh", "Harmonic Filter Thresholds", FrequencyAmpPairConstraint{}),
    FloatPairsArrayParam("pThresh", "Percussive Filter Thresholds", FrequencyAmpPairConstraint{}),
    LongParam("winSize", "Window Size", 1024, Min(4)), LongParam("hopSize", "Hop Size", 512),
    LongParam("fftSize", "FFT Size", -1, LowerLimit<kWinSize>(), PowerOfTwo()),
    LongParam("maxWinSize", "Maxiumm Window Size", 16384) ,
    LongParam("maxHSize", "Maximum Harmonic Filter Size", 101, LowerLimit<kHSize>(), Odd{}),
    LongParam("maxPSize", "Maximum Percussive Filter Size", 101,LowerLimit<kPSize>(), Odd{})
);

using HPSSParamsType = decltype(HPSSParams);

template <typename T, typename U = T> class HPSSClient : public FluidBaseClient<HPSSParamsType>, public AudioIn, public AudioOut
{

  using data_type  = FluidTensorView<T, 2>;
  using complex    = FluidTensorView<std::complex<T>, 1>;
  using HostVector = HostVector<U>;

public:
  HPSSClient():FluidBaseClient<HPSSParamsType>(HPSSParams)
  {
    audioChannelsIn(1);
    audioChannelsOut(3);
  }
  
  HPSSClient(HPSSClient &) = delete;
  HPSSClient operator=(HPSSClient &) = delete;

  // Here we do an STFT and its inverse
//  void process(data_type input, data_type output)
//  {
//    complex spec = mSTFT->processFrame(input.row(0));
//
//    //      mHPSS->setHThreshold(client::lookupParam("hthresh",
//    //      getParams()).getFloat());
//    //      mHPSS->setPThreshold(client::lookupParam("pthresh",
//    //      getParams()).getFloat());
//
//    if (mode() > 0)
//    {
//      mHPSS->setHThresholdX1(client::lookupParam("htf1", getParams()).getFloat());
//      mHPSS->setHThresholdY1(client::lookupParam("hta1", getParams()).getFloat());
//      mHPSS->setHThresholdX2(client::lookupParam("htf2", getParams()).getFloat());
//      mHPSS->setHThresholdY2(client::lookupParam("hta2", getParams()).getFloat());
//    }
//
//    if (mode() == 2)
//    {
//      mHPSS->setPThresholdX1(client::lookupParam("ptf1", getParams()).getFloat());
//      mHPSS->setPThresholdY1(client::lookupParam("pta1", getParams()).getFloat());
//      mHPSS->setPThresholdX2(client::lookupParam("ptf2", getParams()).getFloat());
//      mHPSS->setPThresholdY2(client::lookupParam("pta2", getParams()).getFloat());
//    }
//
//    mHPSS->processFrame(spec, mSeparatedSpectra);
//
//    mHarms = mSeparatedSpectra.col(0);
//    mPerc  = mSeparatedSpectra.col(1);
//    //      mSeparatedSpectra.row(0) = spec;
//    //      mSeparatedSpectra.row(1) = spec;
//
//    output.row(0) = mISTFT->processFrame(mHarms);
//    output.row(1) = mISTFT->processFrame(mPerc);
//    if (mode() == 2)
//    {
//      mResidual     = mSeparatedSpectra.col(2);
//      output.row(2) = mISTFT->processFrame(mResidual);
//    }
//    output.row(3) = mNormWindow;
//  }
//  // Here we gain compensate for the OLA
  size_t latency() { return get<kWinSize>(); }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {

    if (!input[0].data() || !output[0].data()) return;
    // Here we do an STFT and its inverse

    int nBins = get<kFFTSize>() == -1 ? get<kWinSize>() / 2 + 1 : get<kFFTSize>() / 2 + 1;
    if (HPSSNeedsInit(nBins, get<kMaxPSize>(), get<kMaxHSize>()))
    {
        mHPSS.init(nBins, get<kMaxPSize>(), get<kMaxHSize>(), get<kPSize>(), get<kHSize>(),
            get<kMode>(), get<kHThresh>()[0].first, get<kHThresh>()[0].second,
            get<kHThresh>()[1].first, get<kHThresh>()[1].second, get<kPThresh>()[0].first,
            get<kPThresh>()[0].second, get<kPThresh>()[1].first, get<kPThresh>()[0].second);
    }
    else
    {
      mHPSS.setVSize(get<kPSize>());
      if(hSizeChanged(get<kHSize>())) mHPSS.setHSize(get<kHSize>());
      
    }
    
    mSTFTBufferedProcess.process(
        *this, input, output,
        [&](ComplexMatrixView in, ComplexMatrixView out)
        {
            mHPSS.processFrame(in.row(0), out);
        });
  }

private:
  
  bool hSizeChanged(size_t hSize)
  {
    static size_t s{0};
    bool result = s == hSize;
    s = hSize;
    return result;
  }
  
  bool HPSSNeedsInit(int nBins, int maxPSize, int maxHSize)
  {
    static int bins{0};
    static int pSize{0};
    static int hSize{0};
    bool res = (bins != nBins && pSize != maxPSize && hSize != maxHSize);
    bins = nBins;
    pSize = maxPSize;
    hSize = maxHSize;
    return res;
  }

  STFTBufferedProcess<T, U, HPSSClient, kMaxWin, kWinSize, kHopSize, kFFTSize, true> mSTFTBufferedProcess;
  algorithm::RTHPSS mHPSS;
  FluidTensor<T, 1>               mNormWindow;
  FluidTensor<std::complex<T>, 2> mSeparatedSpectra;
  FluidTensor<std::complex<T>, 1> mHarms;
  FluidTensor<std::complex<T>, 1> mPerc;
  FluidTensor<std::complex<T>, 1> mResidual;
//  std::vector<client::Instance>   mParams;
};
} // namespace client
} // namespace fluid

