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
#include "../../data/FluidMemory.hpp"
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
  using ArrayXdMap = Eigen::Map<Eigen::ArrayXd>;

public:
  STFT(index windowSize, index fftSize, index hopSize, index windowType,
       Allocator& alloc)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mMaxWindowSize(windowSize), mWindowBuffer(windowSize, alloc),
        mWindowedFrameBuffer(windowSize, alloc), mFFT(fftSize, alloc)
  {
    ArrayXdMap window(mWindowBuffer.data(), mWindowSize);
    auto windowTypeIndex = static_cast<WindowFuncs::WindowTypes>(windowType);
    WindowFuncs::map()[windowTypeIndex](mWindowSize, window);
  }

  void resize(index windowSize, index fftSize, index hopSize)
  {
    assert(windowSize <= mMaxWindowSize &&
           "STFT: Window Size greater than Max");
    mWindowSize = windowSize;
    mHopSize = hopSize;
    mFrameSize = fftSize / 2 + 1;
    ArrayXdMap window(mWindowBuffer.data(), mWindowSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](mWindowSize, window);
    mFFT.resize(fftSize);
  }

  static void magnitude(const FluidTensorView<std::complex<double>, 2> in,
                        FluidTensorView<double, 2>                     out)
  {
    ArrayXXd mag = _impl::asEigen<Eigen::Array>(in).abs().real();
    out <<= _impl::asFluid(mag);
  }

  static void magnitude(const FluidTensorView<std::complex<double>, 1> in,
                        FluidTensorView<double, 1>                     out)
  {
    ArrayXd mag = _impl::asEigen<Eigen::Array>(in).abs().real();
    out <<= _impl::asFluid(mag);
  }

  static void phase(const FluidTensorView<std::complex<double>, 2> in,
                    FluidTensorView<double, 2>                     out)
  {
    ArrayXXd phase = _impl::asEigen<Eigen::Array>(in).arg().real();
    out <<= _impl::asFluid(phase);
  }

  static void phase(const FluidTensorView<std::complex<double>, 1> in,
                    FluidTensorView<double, 1>                     out)
  {
    phase(FluidTensorView<std::complex<double>, 2>(in),
          FluidTensorView<double, 2>(out));
  }


  void process(const RealVectorView audio, ComplexMatrixView spectrogram)
  {
    index   halfWindow = mWindowSize / 2;
    ArrayXdMap window(mWindowBuffer.data(),mWindowSize);
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
          mFFT.process(padded.segment(i * mHopSize, mWindowSize) * window);
    }
    spectrogram <<= _impl::asFluid(result);
  }

  void processFrame(const RealVectorView frame, ComplexVectorView out)
  {
    assert(frame.size() == mWindowSize);
    ArrayXdMap window(mWindowBuffer.data(), mWindowSize);
    ArrayXdMap windowedFrame(mWindowedFrameBuffer.data(), mWindowSize);
    windowedFrame = _impl::asEigen<Eigen::Array>(frame);
    windowedFrame *= window;
    _impl::asEigen<Eigen::Array>(out) = mFFT.process(windowedFrame);
  }
  
  void processFrame(Eigen::Ref<ArrayXd> frame, Eigen::Ref<ArrayXcd> out)
  {
    assert(frame.size() == mWindowSize);
    ArrayXdMap window(mWindowBuffer.data(), mWindowSize);
    ArrayXdMap windowedFrame(mWindowedFrameBuffer.data(), mWindowSize);
    windowedFrame = frame;
    windowedFrame *= window;
    out = mFFT.process(windowedFrame);
  }

  RealVectorView window()
  {
    return RealVectorView(mWindowBuffer.data(), 0, mWindowSize);
  }

private:
  index   mWindowSize;
  index   mHopSize;
  index   mFrameSize;
  index   mMaxWindowSize;
  rt::vector<double> mWindowBuffer;
  rt::vector<double> mWindowedFrameBuffer;
  FFT     mFFT;
};

class ISTFT
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXcd = Eigen::ArrayXcd;
  using ArrayXXcd = Eigen::ArrayXXcd;
  using ArrayXdMap = Eigen::Map<Eigen::ArrayXd>;

public:
  ISTFT(index windowSize, index fftSize, index hopSize, index windowType,
        Allocator& alloc)
      : mWindowSize(windowSize), mMaxWindowSize(windowSize), mHopSize(hopSize),
        mScale(1 / double(fftSize)), mIFFT(fftSize, alloc),
        mBuffer(mWindowSize, alloc), mWindowBuffer(mWindowSize, alloc)
  {
    ArrayXdMap window(mWindowBuffer.data(), mWindowSize);
    auto windowTypeIndex = static_cast<WindowFuncs::WindowTypes>(windowType);
    WindowFuncs::map()[windowTypeIndex](mWindowSize, window);
  }

  void resize(index windowSize, index fftSize, index hopSize)
  {
    assert(windowSize <= mMaxWindowSize &&
           "STFT: Window Size greater than Max");
    mWindowSize = windowSize;
    mHopSize = hopSize;
    mScale = 1 / double(fftSize);
    mIFFT.resize(fftSize);
  }

  void process(const ComplexMatrixView spectrogram, RealVectorView audio)
  {
    const auto& epsilon = std::numeric_limits<double>::epsilon;
    index       halfWindow = mWindowSize / 2;
    index       nFrames = spectrogram.rows();
    index       outputSize = mWindowSize + (nFrames - 1) * mHopSize;
    outputSize += mWindowSize + mHopSize;
    ArrayXdMap window(mWindowBuffer.data(), mWindowSize);
    ArrayXXcd  specData = _impl::asEigen<Eigen::Array>(spectrogram);
    ArrayXd    outputPadded = ArrayXd::Zero(outputSize);
    ArrayXd    norm = ArrayXd::Zero(outputSize);
    for (index i = 0; i < nFrames; i++)
    {
      ArrayXd frame = mIFFT.process(specData.row(i)).segment(0, mWindowSize);
      outputPadded.segment(i * mHopSize, mWindowSize) +=
          frame * mScale * window;
      norm.segment(i * mHopSize, mWindowSize) += window * window;
    }
    outputPadded = outputPadded / norm.max(epsilon());
    ArrayXd trimmed = outputPadded.segment(halfWindow, audio.size());
    audio <<= _impl::asFluid(trimmed);
  }

  void processFrame(ComplexVectorView frame, RealVectorView audio)
  {
    ArrayXdMap           window(mWindowBuffer.data(), mWindowSize);
    Eigen::Map<ArrayXcd> frameMap(mBuffer.data(), frame.size());
    frameMap = _impl::asEigen<Eigen::Array>(frame);
    _impl::asEigen<Eigen::Array>(audio) =
        mIFFT.process(frameMap) * window * mScale;
  }

  void processFrame(Eigen::Ref<ArrayXcd> frame, Eigen::Ref<ArrayXd> audio)
  {
    ArrayXdMap window(mWindowBuffer.data(), mWindowSize);
    audio = mIFFT.process(frame).segment(0, mWindowSize) * window * mScale;
  }

  RealVectorView window()
  {
    return RealVectorView(mWindowBuffer.data(), 0, mWindowSize);
  }

private:
  index   mWindowSize{1024};
  index   mMaxWindowSize;
  index   mHopSize{512};
  double  mScale{1};
  IFFT    mIFFT;
  rt::vector<std::complex<double>> mBuffer;
  rt::vector<double> mWindowBuffer;
};

} // namespace algorithm
} // namespace fluid
