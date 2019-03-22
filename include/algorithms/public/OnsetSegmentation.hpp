
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/ConvolutionTools.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "OnsetDetectionFuncs.hpp"
#include "WindowFuncs.hpp"
#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>
#include <deque>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::Map;

class OnsetSegmentation {

public:
  OnsetSegmentation(int maxSize)
      : mMaxSize(maxSize), mWindowStorage(maxSize), mFFT(maxSize),
        mFFTSize(1024), mWindowSize(1024), mHopSize(256), mFrameDelta(0),
        mWindowType(WindowTypes::kHann), mFunction(0), mFilterSize(5),
        mThreshold(0.1), mDebounce(2), mDebounceCount(1), mPrevFuncVal(0),
        mFilter(mFilterSize, 0), mSorting(mFilterSize) {
    mFFT.resize(mFFTSize);
    makeWindow();
    initFilter();
  }

  void initFilter() {
    mFilter = std::deque<double>(mFilterSize, 0);
    std::iota(mSorting.begin(), mSorting.end(), 0);
  }

  void sortFilter() {
    for (int i = 1; i < mFilter.size(); ++i) {
      for (int j = i; j > 0 && mFilter[mSorting[j - 1]] > mFilter[mSorting[j]];
           --j) {
        std::swap(mSorting[j - 1], mSorting[j]);
      }
    }
  }

  void makeWindow() {
    mWindowStorage.setZero();
    windows[mWindowType](mWindowSize, mWindowStorage);
    mWindow = mWindowStorage.segment(0, mWindowSize);
    prevFrame = ArrayXcd::Zero(mFFTSize / 2 + 1);
    prevPrevFrame = ArrayXcd::Zero(mFFTSize / 2 + 1);
  }

  void updateParameters(int fftSize, int windowSize, int hopSize,
                        int frameDelta, int function, int filterSize,
                        double threshold, int debounce) {
    assert(fftSize <= mMaxSize);
    assert(windowSize <= mMaxSize);
    assert(windowSize <= fftSize);
    assert(hopSize <= mMaxSize);
    assert(frameDelta <= windowSize);
    assert(filterSize % 2);

    if (fftSize != mFFTSize) {
      mFFTSize = fftSize;
      mFFT.resize(mFFTSize);
      makeWindow();
    }
    if (windowSize != mWindowSize) {
      mWindowSize = windowSize;
      makeWindow();
    }
    mHopSize = hopSize;
    mFrameDelta = frameDelta;
    mFilterSize = filterSize;
    mThreshold = threshold;
    mFunction = function;
    mDebounce = debounce;
  }

  // TODO: review for new version
  void process(const RealVectorView input, RealVectorView output) {
    using algorithm::convolveReal;
    using algorithm::kEdgeWrapCentre;
    int nFrames =
        floor((input.size() + mWindowSize / 2 - mFrameDelta) / mHopSize);
    ArrayXd onsetDetectionFunc(nFrames);
    for (int i = 0; i < nFrames; i++) {
      RealVectorView frame =
          input(fluid::Slice(i * mHopSize, mWindowSize + mFrameDelta));
      onsetDetectionFunc(i) = processFrame(frame);
    }
    if (mFilterSize > 0) {
      ArrayXd filter = ArrayXd::Constant(mFilterSize, 1.0 / mFilterSize);
      ArrayXd smoothed = ArrayXd::Zero(onsetDetectionFunc.size());
      convolveReal(smoothed.data(), onsetDetectionFunc.data(),
                   onsetDetectionFunc.size(), filter.data(), filter.size(),
                   kEdgeWrapCentre);
      onsetDetectionFunc = smoothed;
    }
    onsetDetectionFunc /= onsetDetectionFunc.maxCoeff();
    for (int i = mFilterSize / 2; i < onsetDetectionFunc.size() - 1; i++) {
      if (onsetDetectionFunc(i) > onsetDetectionFunc(i - 1) &&
          onsetDetectionFunc(i) > onsetDetectionFunc(i + 1) &&
          onsetDetectionFunc(i) > mThreshold) {
        output(i - mFilterSize / 2) = 1;
      } else
        output(i - mFilterSize / 2) = 0;
    }
  }

  double processFrame(RealVectorView input) {
    ArrayXd in = asEigen<Array>(input);
    double funcVal = 0;
    double filteredFuncVal = 0;
    double detected = 0.;
    ArrayXcd frame = mFFT.process(in.segment(0, mWindowSize) * mWindow);
    if (mFunction > 1 && mFunction < 5 && mFrameDelta != 0) {
      ArrayXcd frame2 =
          mFFT.process(in.segment(mFrameDelta, mWindowSize) * mWindow);
      funcVal = onsetDetectionFuncs[static_cast<ODF>(mFunction)](frame2, frame,
                                                                 frame);
    } else {
      funcVal = onsetDetectionFuncs[static_cast<ODF>(mFunction)](
          frame, prevFrame, prevPrevFrame);
    }
    filteredFuncVal = funcVal - mFilter[mSorting[(mFilterSize - 1) / 2]];
    mFilter.push_back(funcVal);
    mFilter.pop_front();
    sortFilter();
    prevPrevFrame = prevFrame;
    prevFrame = frame;

    if (filteredFuncVal > mThreshold && mPrevFuncVal < mThreshold &&
        mDebounceCount == 0) {
      detected = 1.0;
      mDebounceCount = mDebounce;
    } else {
      if (mDebounceCount > 0)
        mDebounceCount--;
    }
    mPrevFuncVal = filteredFuncVal;
    return detected;
  }

private:
  FFT mFFT;
  ArrayXd mWindowStorage;
  ArrayXd mWindow;
  int mMaxSize;
  int mFFTSize;
  int mWindowSize;
  int mHopSize;
  int mFrameDelta;
  WindowTypes mWindowType;
  int mFunction;
  int mFilterSize;
  double mThreshold;
  int mDebounce;
  int mDebounceCount;
  std::deque<double> mFilter;
  std::vector<int> mSorting;
  ArrayXcd prevFrame;
  ArrayXcd prevPrevFrame;
  double mPrevFuncVal;
};

}; // namespace algorithm
}; // namespace fluid
