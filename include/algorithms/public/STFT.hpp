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
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class STFT
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using ArrayXcd = Eigen::ArrayXcd;
  using ArrayXXcd = Eigen::ArrayXXcd;

public:
  STFT(index windowSize, index fftSize, index hopSize, index windowType = 0)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mFFT(fftSize)
  {
    mWindow = ArrayXd::Zero(mWindowSize);
    auto windowTypeIndex = static_cast<WindowFuncs::WindowTypes>(windowType);
    WindowFuncs::map()[windowTypeIndex](mWindowSize, mWindow);
  }

  static void magnitude(const FluidTensorView<std::complex<double>, 2> in,
                        FluidTensorView<double, 2>                      out)
  {
    ArrayXXd mag = _impl::asEigen<Eigen::Array>(in).abs().real();
    out = _impl::asFluid(mag);
  }

  static void magnitude(const FluidTensorView<std::complex<double>, 1> in,
                        FluidTensorView<double, 1>                      out)
  {
    ArrayXd mag = _impl::asEigen<Eigen::Array>(in).abs().real();
    out = _impl::asFluid(mag);
  }

  static void phase(const FluidTensorView<std::complex<double>, 2> in,
                        FluidTensorView<double, 2>                      out)
  {
    ArrayXXd phase = _impl::asEigen<Eigen::Array>(in).arg().real();
    out = _impl::asFluid(phase);
  }
  
  static void phase(const FluidTensorView<std::complex<double>, 1> in,
                        FluidTensorView<double, 1>                  out)
  {    
    phase(FluidTensorView<std::complex<double>,2>(in),
          FluidTensorView<double,2>(out)); 
  }


  void process(const RealVectorView audio, ComplexMatrixView spectrogram)
  {
    index   halfWindow = mWindowSize / 2;
    ArrayXd padded(audio.size() + mWindowSize + mHopSize);
    padded.fill(0);
    padded.segment(halfWindow, audio.size()) =
        Eigen::Map<const ArrayXd>(audio.data(), audio.size());
    index nFrames = static_cast<index>(
        std::floor((padded.size() - mWindowSize) / mHopSize));

    ArrayXXcd result(nFrames, mFrameSize);
    for (index i = 0; i < nFrames; i++)
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

  void processFrame(Eigen::Ref<ArrayXd> frame, Eigen::Ref<ArrayXcd> out)
  {
    assert(frame.size() == mWindowSize);
    out = mFFT.process(frame * mWindow);
  }


  RealVectorView window()
  {
    return RealVectorView(mWindow.data(), 0, mWindowSize);
  }

private:
  index   mWindowSize;
  index   mHopSize;
  index   mFrameSize;
  ArrayXd mWindow;
  FFT     mFFT;
};

class ISTFT
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXcd = Eigen::ArrayXcd;
  using ArrayXXcd = Eigen::ArrayXXcd;

public:
  ISTFT(index windowSize, index fftSize, index hopSize, index windowType = 0)
      : mWindowSize(windowSize), mHopSize(hopSize), mScale(1 / double(fftSize)),
        mIFFT(fftSize), mBuffer(mWindowSize)
  {
    mWindow = ArrayXd::Zero(mWindowSize);
    auto windowTypeIndex = static_cast<WindowFuncs::WindowTypes>(windowType);
    WindowFuncs::map()[windowTypeIndex](mWindowSize, mWindow);
    mWindowSquared = mWindow * mWindow;
  }

  void process(const ComplexMatrixView spectrogram, RealVectorView audio)
  {
    const auto& epsilon = std::numeric_limits<double>::epsilon;

    index halfWindow = mWindowSize / 2;
    index nFrames = spectrogram.rows();
    index outputSize = mWindowSize + (nFrames - 1) * mHopSize;
    outputSize += mWindowSize + mHopSize;
    ArrayXXcd specData = _impl::asEigen<Eigen::Array>(spectrogram);
    ArrayXd   outputPadded = ArrayXd::Zero(outputSize);
    ArrayXd   norm = ArrayXd::Zero(outputSize);
    for (index i = 0; i < nFrames; i++)
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
    mBuffer = mIFFT.process(_impl::asEigen<Eigen::Array>(frame))
                  .segment(0, mWindowSize) *
              mWindow * mScale;
    audio = _impl::asFluid(mBuffer);
  }

  void processFrame(Eigen::Ref<ArrayXcd> frame, Eigen::Ref<ArrayXd> audio)
  {
    audio = mIFFT.process(frame).segment(0, mWindowSize) * mWindow * mScale;
  }

  RealVectorView window()
  {
    return RealVectorView(mWindow.data(), 0, mWindowSize);
  }

private:
  index   mWindowSize{1024};
  index   mHopSize{512};
  ArrayXd mWindow;
  ArrayXd mWindowSquared;
  double  mScale{1};
  IFFT    mIFFT;
  ArrayXd mBuffer;
};

} // namespace algorithm
} // namespace fluid
