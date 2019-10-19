/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "WindowFuncs.hpp"

#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class STFT
{


  using ArrayXd   = Eigen::ArrayXd;
  using ArrayXXd  = Eigen::ArrayXXd;
  using ArrayXcd  = Eigen::ArrayXcd;
  using ArrayXXcd = Eigen::ArrayXXcd;

public:
  STFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mFFT(fftSize)
  {
    mWindow = ArrayXd::Zero(mWindowSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](mWindowSize, mWindow);
  }

  static void magnitude(const FluidTensorView<complex<double>, 2>& in,
                        FluidTensorView<double, 2>                 out)
  {
    ArrayXXd mag = _impl::asEigen<Eigen::Array>(in).abs().real();
    out = _impl::asFluid(mag);
  }

  static void magnitude(const FluidTensorView<complex<double>, 1>& in,
                        FluidTensorView<double, 1>                 out)
  {
    ArrayXd mag = _impl::asEigen<Eigen::Array>(in).abs().real();
    out = _impl::asFluid(mag);
  }


  void process(const RealVectorView audio, ComplexMatrixView spectrogram)
  {
    int     halfWindow = mWindowSize / 2;
    ArrayXd padded(audio.size() + mWindowSize + mHopSize);
    padded.fill(0);
    padded.segment(halfWindow, audio.size()) =
        Eigen::Map<const ArrayXd>(audio.data(), audio.size());
    int       nFrames = floor((padded.size() - mWindowSize) / mHopSize);
    ArrayXXcd result(nFrames, mFrameSize);
    for (int i = 0; i < nFrames; i++)
    {
      result.row(i) =
          mFFT.process(padded.segment(i * mHopSize, mWindowSize) * mWindow);
    }
    spectrogram = _impl::asFluid(result);
  }

  void processFrame(const RealVectorView frame, ComplexVectorView out)
  {
    assert(frame.size() == mWindowSize);
    ArrayXcd spectrum =
        mFFT.process(_impl::asEigen<Eigen::Array>(frame) * mWindow);
    out = _impl::asFluid(spectrum);
  }

  RealVectorView window()
  {
    return RealVectorView(mWindow.data(), 0, mWindowSize);
  }

private:
  size_t  mWindowSize;
  size_t  mHopSize;
  size_t  mFrameSize;
  ArrayXd mWindow;
  FFT     mFFT;
};

class ISTFT
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXcd = Eigen::ArrayXXcd;

public:
  ISTFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mScale(1 / double(fftSize)), mIFFT(fftSize), mBuffer(mWindowSize)
  {
    mWindow = ArrayXd::Zero(mWindowSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](mWindowSize, mWindow);
    mWindowSquared = mWindow * mWindow;
  }

  void process(const ComplexMatrixView spectrogram, RealVectorView audio)
  {
    const auto& epsilon = std::numeric_limits<double>::epsilon;

    int halfWindow = mWindowSize / 2;
    int nFrames = spectrogram.rows();
    int outputSize = mWindowSize + (nFrames - 1) * mHopSize;
    outputSize += mWindowSize + mHopSize;
    ArrayXXcd specData = _impl::asEigen<Eigen::Array>(spectrogram);
    ArrayXd   outputPadded = ArrayXd::Zero(outputSize);
    ArrayXd   norm = ArrayXd::Zero(outputSize);
    for (int i = 0; i < nFrames; i++)
    {
      ArrayXd frame = mIFFT.process(specData.row(i)).segment(0, mWindowSize);
      outputPadded.segment(i * mHopSize, mWindowSize) +=
          frame * mScale * mWindow;
      norm.segment(i * mHopSize, mWindowSize) += mWindow * mWindow;
    }
    outputPadded = outputPadded / norm.max(epsilon());
    ArrayXd trimmed = outputPadded.segment(halfWindow, audio.size());
    audio = _impl::asFluid(trimmed);
  }

  void processFrame(const ComplexVectorView frame, RealVectorView audio)
  {
    assert(frame.size() == mFrameSize);
    mBuffer = mIFFT.process(_impl::asEigen<Eigen::Array>(frame))
                  .segment(0, mWindowSize) *
              mWindow * mScale;
    audio = _impl::asFluid(mBuffer);
  }

  RealVectorView window()
  {
    return RealVectorView(mWindow.data(), 0, mWindowSize);
  }

private:
  size_t  mWindowSize{1024};
  size_t  mHopSize{512};
  size_t  mFrameSize{513};
  ArrayXd mWindow;
  ArrayXd mWindowSquared;
  double  mScale{1};
  IFFT    mIFFT;
  ArrayXd mBuffer;
};

} // namespace algorithm
} // namespace fluid
