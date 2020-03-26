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

#include "WindowFuncs.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/ConvolutionTools.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PartialTracking.hpp"
#include "../util/PeakDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>
#include <queue>

namespace fluid {
namespace algorithm {


class SineExtraction
{
  using ArrayXd = Eigen::ArrayXd;
  using VectorXd = Eigen::VectorXd;
  using ArrayXcd = Eigen::ArrayXcd;
  template <typename T>
  using vector = std::vector<T>;

public:
  void init(index windowSize, index fftSize, index transformSize)
  {
    mBins = fftSize / 2 + 1;
    mCurrentFrame = 0;
    mBuf = std::queue<ArrayXcd>();
    mScale = 1.0 / (windowSize / 4.0); // scale to original amplitude
    computeWindowTransform(windowSize, transformSize);
    mTracking.init();
    mWindowBinIncr = mWindowTransform.size() / (mBins - 1) / 2;
    mInvWindowBinIncr = 1.0 / mWindowBinIncr;
    mInitialized = true;
  }

  void processFrame(const ComplexVectorView in, ComplexMatrixView out,
                    double sampleRate, double detectionThreshold,
                    index minTrackLength, double birthLowThreshold,
                    double birthHighThreshold, index trackMethod, double zetaA,
                    double zetaF, double delta, index bandwidth)
  {
    assert(mInitialized);
    using namespace Eigen;
    index    fftSize = 2 * (mBins - 1);
    ArrayXcd frame = _impl::asEigen<Array>(in);
    if (minTrackLength != mTracking.minTrackLength())
    { mBuf = std::queue<ArrayXcd>(); }
    mBuf.push(frame);
    ArrayXd mag = frame.abs().real();
    mag = mag * mScale;
    ArrayXd          logMag = 20 * mag.max(epsilon).log10();
    vector<SinePeak> peaks;
    auto tmpPeaks = mPeakDetection.process(logMag, 0, -infinity, true, false);
    for (auto p : tmpPeaks)
    {
      if (p.second > detectionThreshold)
      {
        double hz = sampleRate * p.first / fftSize;
        peaks.push_back({hz, p.second, false});
      }
    }
    double maxAmp = 20 * std::log10(mag.maxCoeff());
    mTracking.processFrame(peaks, maxAmp, minTrackLength, birthLowThreshold,
                           birthHighThreshold, trackMethod, zetaA, zetaF,
                           delta);
    vector<SinePeak> sinePeaks = mTracking.getActivePeaks();
    ArrayXd          frameSines = ArrayXd::Zero(mBins);
    for (auto& p : sinePeaks)
    { frameSines += synthesizePeak(p, sampleRate, bandwidth); }
    ArrayXXcd result(mBins, 2);
    if (asSigned(mBuf.size()) <= mTracking.minTrackLength())
    {
      result.col(0) = ArrayXd::Zero(mBins);
      result.col(1) = ArrayXd::Zero(mBins);
    }
    else
    {
      ArrayXcd resultFrame = mBuf.front();
      ArrayXd  resultMag = resultFrame.abs().real();
      for (index i = 0; i < mBins; i++)
      {
        if (frameSines(i) >= resultMag(i))
        {
          result(i, 0) = resultFrame(i);
          result(i, 1) = 0;
        }
        else
        {
          double sineWeight = frameSines(i) / resultMag(i);
          result(i, 0) = resultFrame(i) * sineWeight;
          result(i, 1) = resultFrame(i) * (1 - sineWeight);
        }
      }
      mBuf.pop();
    }
    mTracking.prune();
    out = _impl::asFluid(result);
    mCurrentFrame++;
  }


  void reset() { mCurrentFrame = 0; }

  bool initialized() { return mInitialized; }

private:
  void computeWindowTransform(index windowSize, index transformSize)
  {
    index halfBW = transformSize / 2;
    mWindowTransform = ArrayXd::Zero(transformSize);
    ArrayXd window = ArrayXd::Zero(windowSize);
    FFT     fft(transformSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](windowSize, window);
    ArrayXcd transform =
        fft.process(Eigen::Map<ArrayXd>(window.data(), windowSize));
    for (index i = 0; i < halfBW; i++)
    {
      mWindowTransform(halfBW + i) = mWindowTransform(halfBW - i) =
          std::abs(transform(i));
    }
  }

  double interpolateWindow(double pos)
  {
    index  floor = std::lrint(std::floor(pos));
    double frac = pos - floor;
    double dY = mWindowTransform(floor + 1) - mWindowTransform(floor);
    return mWindowTransform(floor) + frac * mInvWindowBinIncr * dY;
  }

  ArrayXd synthesizePeak(SinePeak p, double sampleRate, index bandwidth)
  {
    using namespace std;
    index   halfBW = bandwidth / 2;
    ArrayXd sine = ArrayXd::Zero(mBins);
    double  freqBin = p.freq * 2 * (mBins - 1) / sampleRate;
    if (freqBin >= mBins - 1) freqBin = mBins - 1;
    if (freqBin < 0) freqBin = 0;
    index  freqBinFloor = lrint(floor(freqBin));
    index  freqBinCeil = freqBinFloor + 1;
    double amp = 0.5 * pow(10, p.logMag / 20);
    double pos = mWindowTransform.size() / 2 +
                 ((freqBinCeil - freqBin) * mWindowBinIncr);
    for (index i = freqBinCeil; pos < mWindowTransform.size() - 2 &&
                                i < min(freqBinCeil + halfBW, mBins - 1);
         i++, pos += mWindowBinIncr)
    { sine[i] = amp * interpolateWindow(pos); }
    pos = (mWindowTransform.size() / 2) -
          ((freqBin - freqBinFloor) * mWindowBinIncr);
    for (index i = freqBinFloor;
         pos > 1 && i > max(freqBinFloor - halfBW, asSigned(0));
         i--, pos -= mWindowBinIncr)
    { sine[i] = amp * interpolateWindow(pos); }
    return sine;
  }

  PeakDetection        mPeakDetection;
  PartialTracking      mTracking;
  index                mBins{513};
  index                mCurrentFrame{0};
  std::queue<ArrayXcd> mBuf;
  ArrayXd              mWindowTransform;
  double               mScale{1.0};
  bool                 mInitialized{false};
  double               mWindowBinIncr;
  double               mInvWindowBinIncr;
};
} // namespace algorithm
} // namespace fluid
