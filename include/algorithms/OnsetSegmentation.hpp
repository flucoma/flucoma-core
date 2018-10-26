
#pragma once

#include "Descriptors.hpp"
#include "FFT.hpp"
#include "Windows.hpp"
#include "algorithms/ConvolutionTools.hpp"
#include <Eigen/Eigen>
#include <algorithm>

namespace fluid {
namespace algorithm {

using Eigen::Map;

class OnsetSegmentation {
  using RealVector = fluid::FluidTensor<double, 1>;
  using RealVectorView = fluid::FluidTensorView<double, 1>;
  using WindowType = algorithm::WindowType;
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXcd = Eigen::ArrayXcd;
  using ArrayXdMap = Map<Eigen::Array<double, Eigen::Dynamic, Eigen::RowMajor>>;
  using FFT = algorithm::FFT;

public:
  enum Normalisation {
    kNone,
    kAmplitude,
    kPower,
  };

  enum DifferenceFunction {
    kL1Norm,
    kL2Norm,
    kLogDifference,
    kFoote,
    kItakuraSaito,
    kKullbackLiebler,
    kSymmetricKullbackLiebler,
    kModifiedKullbackLiebler
  };

  OnsetSegmentation(int FFTSize, int windowSize, int hopSize, int frameDelta,
                    WindowType windowType, double threshold,
                    DifferenceFunction function, int filterSize,
                    bool forwardOnly = false)
      : mFFT(FFTSize), mFFTSize(FFTSize), mWindowSize(windowSize),
        mHopSize(hopSize), mFrameDelta(frameDelta),
        mWindowType(algorithm::WindowType::kHann), mFunction(function),
        mFilterSize(filterSize), mForwardOnly(forwardOnly),
        mNormalisation(kNone), mThreshold(threshold) {
    assert(mWindowSize <= mFFTSize);
    assert(mFrameDelta <= mWindowSize);
    resizeStorage();
  }

  void setParameters(DifferenceFunction function, bool forwardOnly,
                     Normalisation normalisation) {
    mFunction = function;
    mForwardOnly = forwardOnly;
    mNormalisation = normalisation;
  }

  int FFTSize() const { return mFFTSize; }
  int windowSize() const { return mWindowSize; }
  int frameDelta() const { return mFrameDelta; }
  int inputFrameSize() const { return windowSize() + frameDelta(); }
  int nFrames(int inputSize) const {
    return floor(
        (inputSize + inputFrameSize() / 2 + mWindowSize - inputFrameSize()) /
        mHopSize);
  }

  void process(RealVectorView &input, RealVectorView &output) {
    using algorithm::convolveReal;
    using algorithm::kEdgeWrapCentre;
    int frameSize = inputFrameSize();
    int leftPadding = frameSize / 2;
    int rightPadding = mWindowSize;
    ArrayXd padded(input.size() + leftPadding + rightPadding);
    padded.fill(0);
    padded.segment(leftPadding, input.size()) =
        Map<const ArrayXd>(input.data(), input.size());
    int nFrames = floor((padded.size() - frameSize) / mHopSize);
    ArrayXd onsetDetectionFunc(nFrames);
    for (int i = 0; i < nFrames; i++) {
      RealVectorView frame = input(fluid::Slice(i * mHopSize, frameSize));
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
    for (int i = 1; i < onsetDetectionFunc.size() - 1; i++) {
      if (onsetDetectionFunc(i) > onsetDetectionFunc(i - 1) &&
          onsetDetectionFunc(i) > onsetDetectionFunc(i + 1) &&
          onsetDetectionFunc(i) > mThreshold && i > mFilterSize / 2) {
        output(i - mFilterSize / 2) = 1;
      } else
        output(i - mFilterSize / 2) = 0;
    }
  }

  double processFrame(RealVectorView &input) {
    processSingleWindow(mFrame1, input.data() + frameDelta());
    processSingleWindow(mFrame2, input.data());
    RealVectorView frame1View(mFrame1);
    RealVectorView frame2View(mFrame2);
    return frameComparison(frame1View, frame2View);
  }

private:
  void processSingleWindow(RealVector &frame, const double *input) {
    for (auto i = 0; i < frame.size(); i++)
      mFFTBuffer(i) = input[i];

    mFFTBuffer *= mWindow;
    Eigen::Ref<ArrayXcd> fftOut = mFFT.process(Eigen::Ref<ArrayXd>(mFFTBuffer));

    for (auto i = 0; i < frame.size(); i++) {
      const double real = fftOut(i).real();
      const double imag = fftOut(i).imag();
      frame(i) = sqrt(real * real + imag * imag);
    }
  }

  void clipEpsilon(RealVectorView &input) {
    for (auto it = input.begin(); it != input.end(); it++)
      *it = std::max(std::numeric_limits<double>::epsilon(), *it);
  }

  double frameComparison(RealVectorView &vec1, RealVectorView &vec2)
  {
    if (mForwardOnly)
      Descriptors::forwardFilter(vec1, vec2);

    if (mNormalisation != kNone) {
      Descriptors::normalise(vec1, mNormalisation == kPower);
      Descriptors::normalise(vec2, mNormalisation == kPower);
    }

    // TODO - review this later

    clipEpsilon(vec1);
    clipEpsilon(vec2);

    switch (mFunction) {
    case kL1Norm:
      return Descriptors::differenceL1Norm(vec1, vec2);
    case kL2Norm:
      return Descriptors::differenceL2Norm(vec1, vec2);
    case kLogDifference:
      return Descriptors::differenceLog(vec1, vec2);
    case kFoote:
      return Descriptors::differenceFT(vec1, vec2);
    case kItakuraSaito:
      return Descriptors::differenceIS(vec1, vec2);
    case kKullbackLiebler:
      return Descriptors::differenceKL(vec1, vec2);
    case kSymmetricKullbackLiebler:
      return Descriptors::differenceSKL(vec1, vec2);
    case kModifiedKullbackLiebler:
      return Descriptors::differenceMKL(vec1, vec2);
    }
  }

  void resizeStorage() {
    int FFTFrameSize = FFTSize() / 2 + 1;
    mFrame1.resize(FFTFrameSize);
    mFrame2.resize(FFTFrameSize);
    mFFTBuffer.resize(windowSize());

    mWindow = Map<ArrayXd>(
        algorithm::windowFuncs[mWindowType](windowSize()).data(), windowSize());
  }

private:
  FFT mFFT;
  RealVector mFrame1;
  RealVector mFrame2;
  ArrayXd mFFTBuffer;
  ArrayXd mWindow;

  int mFFTSize;
  int mWindowSize;
  int mHopSize;
  int mFrameDelta;
  WindowType mWindowType;
  double mThreshold;
  int mFilterSize;

  DifferenceFunction mFunction;
  bool mForwardOnly;
  Normalisation mNormalisation;
};

}; // namespace algorithm
}; // namespace fluid
