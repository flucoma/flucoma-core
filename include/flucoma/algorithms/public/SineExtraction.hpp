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

#include "WindowFuncs.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PartialTracking.hpp"
#include "../util/PeakDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
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
  using vector = rt::vector<T>;

  using Queue = rt::queue<ScopedEigenMap<ArrayXcd>>;

  Queue makeEmptyQueue(Allocator& alloc)
  {
    return Queue{rt::deque<ScopedEigenMap<ArrayXcd>>(alloc)};
  }


public:
  SineExtraction(index maxFFT, Allocator& alloc)
      : mTracking(alloc), mBuf{makeEmptyQueue(alloc)},
        mWindowTransform(maxFFT, alloc)
  {}

  void init(index windowSize, index fftSize, index transformSize,
            Allocator& alloc)
  {
    mBins = fftSize / 2 + 1;
    mCurrentFrame = 0;
    mBuf = makeEmptyQueue(alloc);
    //    mBuf = std::queue<ArrayXcd>();
    mScale = 1.0 / (windowSize / 4.0); // scale to original amplitude
    computeWindowTransform(windowSize, transformSize, alloc);
    mTracking.init();
    mWindowBinIncr = mWindowTransform.size() / (mBins - 1) / 2;
    mInvWindowBinIncr = 1.0 / mWindowBinIncr;
    mInitialized = true;
  }

  void processFrame(const ComplexVectorView in, ComplexMatrixView out,
                    double sampleRate, double detectionThreshold,
                    index minTrackLength, double birthLowThreshold,
                    double birthHighThreshold, index trackMethod, double zetaA,
                    double zetaF, double delta, index bandwidth,
                    Allocator& alloc)
  {
    assert(mInitialized);
    using namespace Eigen;
    index                    fftSize = 2 * (mBins - 1);
    ScopedEigenMap<ArrayXcd> frame(in.size(), alloc);
    frame = _impl::asEigen<Array>(in);

    if (minTrackLength != mTracking.minTrackLength())
    {
      mBuf = makeEmptyQueue(alloc);
    }

    mBuf.push(frame);
    ScopedEigenMap<ArrayXd> mag(in.size(), alloc);
    mag = frame.abs().real();
    mag = mag * mScale;
    ScopedEigenMap<ArrayXd> logMag(in.size(), alloc);
    logMag = 20 * mag.max(epsilon).log10();

    vector<SinePeak> peaks(0, alloc);
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
                           birthHighThreshold, trackMethod, zetaA, zetaF, delta,
                           alloc);
    vector<SinePeak>        sinePeaks = mTracking.getActivePeaks(alloc);
    ScopedEigenMap<ArrayXd> frameSines(mBins, alloc);
    frameSines.setZero();
    for (auto& p : sinePeaks)
    {
      frameSines += synthesizePeak(p, sampleRate, bandwidth, alloc);
    }
    ScopedEigenMap<ArrayXXcd> result(mBins, 2, alloc);
    if (asSigned(mBuf.size()) <= mTracking.minTrackLength())
    {
      result.col(0).setZero();
      result.col(1).setZero();
    }
    else
    {
      ScopedEigenMap<ArrayXcd>& resultFrame = mBuf.front();
      ScopedEigenMap<ArrayXd>   resultMag(resultFrame.size(), alloc);
      resultMag = resultFrame.abs().real();
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
    _impl::asEigen<Array>(out) = result;
    mCurrentFrame++;
  }

  void reset() { mCurrentFrame = 0; }

  bool initialized() const { return mInitialized; }

private:
  void computeWindowTransform(index windowSize, index transformSize,
                              Allocator& alloc)
  {
    index halfBW = transformSize / 2;
    mWindowTransform.head(transformSize) = ArrayXd::Zero(transformSize);
    ScopedEigenMap<ArrayXd> window(ArrayXd::Zero(windowSize), alloc);
    FFT                     fft(transformSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](windowSize, window);
    ScopedEigenMap<ArrayXcd> transform(transformSize / 2 + 1, alloc);
    transform = fft.process(Eigen::Map<ArrayXd>(window.data(), windowSize));
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

  ScopedEigenMap<ArrayXd> synthesizePeak(SinePeak p, double sampleRate,
                                         index bandwidth, Allocator& alloc)
  {
    using namespace std;
    index                   halfBW = bandwidth / 2;
    ScopedEigenMap<ArrayXd> sine(mBins, alloc);
    double                  freqBin = p.freq * 2 * (mBins - 1) / sampleRate;
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
    {
      sine[i] = amp * interpolateWindow(pos);
    }
    pos = (mWindowTransform.size() / 2) -
          ((freqBin - freqBinFloor) * mWindowBinIncr);
    for (index i = freqBinFloor;
         pos > 1 && i > max(freqBinFloor - halfBW, asSigned(0));
         i--, pos -= mWindowBinIncr)
    {
      sine[i] = amp * interpolateWindow(pos);
    }
    return sine;
  }

  PeakDetection           mPeakDetection;
  PartialTracking         mTracking;
  index                   mBins{513};
  index                   mCurrentFrame{0};
  Queue                   mBuf;
  ScopedEigenMap<ArrayXd> mWindowTransform;
  double                  mScale{1.0};
  bool                    mInitialized{false};
  double                  mWindowBinIncr;
  double                  mInvWindowBinIncr;
};
} // namespace algorithm
} // namespace fluid
