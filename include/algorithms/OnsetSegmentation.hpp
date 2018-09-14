
#pragma once

#include "Descriptors.hpp"
#include "FFT.hpp"
#include "Windows.hpp"
#include <Eigen/Eigen>
#include <algorithm>

namespace fluid {
namespace segmentation {
  
using Eigen::Map;

class OnsetSegmentation : public fft::FFT
{
  using RealTensor = fluid::FluidTensor<double, 1>;
  using Real = fluid::FluidTensorView<double, 1>;
  using WindowType = windows::WindowType;
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXcd = Eigen::ArrayXcd;

public:

  enum Normalisation
  {
    kNone,
    kAmplitude,
    kPower,
  };
  
  enum DifferenceFunction
  {
    kL1Norm,
    kL2Norm,
    kLogDifference,
    kFoote,
    kItakuraSaito,
    kKullbackLiebler,
    kSymmetricKullbackLiebler,
    kModifiedKullbackLiebler
  };
  
  OnsetSegmentation(int FFTSize) : fft::FFT(FFTSize), mFFTSize(FFTSize), mWindowSize(0), mFrameDelta(0), mWindowType(windows::WindowType::Hann), mFunction(kL1Norm), mForwardOnly(false), mNormalisation(kNone)
  {
    resizeStorage();
  }

  void setParameters(DifferenceFunction function, bool forwardOnly, Normalisation normalisation)
  {
    mFunction = function;
    mForwardOnly = forwardOnly;
    mNormalisation = normalisation;
  }
  
  void prepareStream(int windowSize, int frameDelta, WindowType windowType)
  {
    mWindowSize = std::min(mWindowSize, mFFTSize);
    mFrameDelta = frameDelta;
    mWindowType = windowType;
    resizeStorage();
  }
  
  int FFTSize() const         { return mFFTSize; }
  int windowSize() const      { return mWindowSize; }
  int frameDelta() const      { return mFrameDelta; }
  int inputFrameSize() const  { return windowSize() + frameDelta(); }
  
private:

  double processFrame(const double *input)
  {
    processSingleWindow(mFrame1, input + frameDelta());
    processSingleWindow(mFrame2, input);
    
    Real frame1View(mFrame1);
    Real frame2View(mFrame2);

    return frameComparison(frame1View, frame2View);
  }
  
  void processSingleWindow(RealTensor& frame, const double *input)
  {
    for (auto i = 0; i < frame.size(); i++)
      mFFTBuffer(i) = input[i];
    
    mFFTBuffer *= mWindow;
    Eigen::Ref<ArrayXcd> fftOut = FFT::process(Eigen::Ref<ArrayXd>(mFFTBuffer));
    
    for (auto i = 0; i < frame.size(); i++)
    {
      const double real = fftOut(i).real();
      const double imag = fftOut(i).imag();
      frame(i) = sqrt(real * real + imag * imag);
    }
  }
  
  void clipEpsilon(Real& input)
  {
    for (auto it = input.begin(); it != input.end(); it++)
      *it = std::max(std::numeric_limits<double>::epsilon(), *it);
  }
  
  double frameComparison(Real& vec1, Real& vec2)
  {
    using namespace descriptors;
    
    if (mForwardOnly)
      Descriptors::forwardFilter(vec1, vec2);
    
    if (mNormalisation != kNone)
    {
      Descriptors::normalise(vec1, mNormalisation == kPower);
      Descriptors::normalise(vec2, mNormalisation == kPower);
    }
    
    // TODO - review this later
    
    clipEpsilon(vec1);
    clipEpsilon(vec2);
    
    switch (mFunction)
    {
      case kL1Norm:                      return Descriptors::differenceL1Norm(vec1, vec2);
      case kL2Norm:                      return Descriptors::differenceL2Norm(vec1, vec2);
      case kLogDifference:               return Descriptors::differenceLog(vec1, vec2);
      case kFoote:                       return Descriptors::differenceFT(vec1, vec2);
      case kItakuraSaito:                return Descriptors::differenceIS(vec1, vec2);
      case kKullbackLiebler:             return Descriptors::differenceKL(vec1, vec2);
      case kSymmetricKullbackLiebler:    return Descriptors::differenceSKL(vec1, vec2);
      case kModifiedKullbackLiebler:     return Descriptors::differenceMKL(vec1, vec2);
    }
  }
  
  void resizeStorage()
  {
    int FFTFrameSize = FFTSize() / 2 + 1;
    mFrame1.resize(FFTFrameSize);
    mFrame2.resize(FFTFrameSize);
    mFFTBuffer.resize(windowSize());
    
    mWindow = Map<ArrayXd>(windows::windowFuncs[mWindowType](windowSize()).data(), windowSize());
  }
  
private:

  RealTensor mFrame1;
  RealTensor mFrame2;
  ArrayXd mFFTBuffer;
  ArrayXd mWindow;
  
  int mFFTSize;
  int mWindowSize;
  int mFrameDelta;
  WindowType mWindowType;
  
  DifferenceFunction mFunction;
  bool mForwardOnly;
  Normalisation mNormalisation;
};

};  // namespace segmentation
};  // namespace fluid
