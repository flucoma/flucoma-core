#pragma once

#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>

#include <algorithms/public/STFT.hpp>
#include <clients/common/FluidSink.hpp>
#include <clients/common/FluidSource.hpp>

namespace fluid {
namespace client {

class BufferedProcess {
public:
  template <typename F>
  void process(std::size_t windowSize, std::size_t hopSize, F processFunc) {

    assert(windowSize <= maxWindowSize() && "Window bigger than maximum");
    for (; mFrameTime < mHostSize; mFrameTime += hopSize) {
      if (mSource) {
        RealMatrix windowIn = mFrameIn(Slice(0), Slice(0, windowSize));
        FluidTensorView<double, 2> windowOut =
            mFrameOut(Slice(0), Slice(0, windowSize));
        mSource->pull(windowIn, mFrameTime);
        processFunc(windowIn, windowOut);
        if (mSink)
          mSink->push(windowOut, mFrameTime);
      }
      
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  std::size_t hostSize() const noexcept { return mHostSize; }
  void hostSize(std::size_t size) noexcept { mHostSize = size; }
  
  std::size_t maxWindowSize() const noexcept { return mFrameIn.cols(); }
  void maxSize(std::size_t frames, std::size_t channelsIn,
               std::size_t channelsOut) {
    if (channelsIn > mFrameIn.rows() || frames > mFrameIn.cols())
      mFrameIn.resize(channelsIn, frames);
    if (channelsOut > mFrameOut.rows() || frames > mFrameOut.cols())
      mFrameOut.resize(channelsOut, frames);
  }

  void setBuffers(FluidSource<double> &source, FluidSink<double> &sink) {
    mSource = &source;
    mSink = &sink;
  }

private:
  std::size_t mFrameTime;
  std::size_t mHostSize;
  FluidTensor<double, 2> mFrameIn;
  FluidTensor<double, 2> mFrameOut;
  FluidSource<double> *mSource = nullptr;
  FluidSink<double> *mSink = nullptr;
};

template <typename T, typename U, typename Client, size_t maxWinParam,
          size_t winParam, size_t hopParam, size_t FFTParam, bool normalise>
class STFTBufferedProcess {
  using HostVector = HostVector<U>;

public:
  template <typename F>
  void process(Client &x, std::vector<HostVector> &input,
               std::vector<HostVector> &output, F &&processFunc) {

    size_t winSize = x.template get<winParam>();
    size_t hopSize = x.template changed<hopParam>() ? x.template get<hopParam>()
                                                    : winSize / 2;
    size_t fftSize =
        x.template changed<FFTParam>() ? x.template get<FFTParam>() : winSize;

    // TODO: constraints check here: error and bail if unmet

    bool newParams = paramsChanged(winSize, hopSize, fftSize);

    if (!mSTFT.get() || newParams)
      mSTFT.reset(new algorithm::STFT(winSize, fftSize, hopSize));

    if (!mISTFT.get() || newParams)
      mISTFT.reset(new algorithm::ISTFT(winSize, fftSize, hopSize));

    if (!input[0].data())
      return; // if there's not actually an audio input, no point continuing

    std::size_t chansIn = x.audioChannelsIn();
    std::size_t chansOut = x.audioChannelsOut();

    size_t hostBufferSize = input[0].size();
    mBufferedProcess.hostSize(hostBufferSize); // safe assumption?
    mInputBuffer.setHostBufferSize(hostBufferSize);
    mOutputBuffer.setHostBufferSize(hostBufferSize);

    std::size_t maxWin = x.template get<maxWinParam>();
    mInputBuffer.setSize(maxWin);
    mOutputBuffer.setSize(maxWin);
    mInputBuffer.reset(chansIn);
    // TODO: make explicit the extra channel for post-normalisation
    mOutputBuffer.reset(chansOut + 1);
    mBufferedProcess.maxSize(maxWin, chansIn, chansOut + 1);

    if (maxWin > mFrameAndWindow.cols())
      mFrameAndWindow.resize(chansOut + 1, maxWin);
    if ((fftSize / 2 + 1) != mSpectrogram.cols())
      mSpectrogram.resize(2, (fftSize / 2 + 1));

    mBufferedProcess.setBuffers(mInputBuffer, mOutputBuffer);

    mInputBuffer.push(HostMatrix<U>(input[0]));

    mBufferedProcess.process(
        winSize, hopSize,
        [this, &processFunc](RealMatrix &in, RealMatrix &out) {
          mSTFT->processFrame(in.row(0), mSpectrogram.row(0));
          processFunc(mSpectrogram.row(0), mSpectrogram.row(1));
          mISTFT->processFrame(mSpectrogram.row(1), out.row(0));
          out.row(1) = mSTFT->window();
          out.row(1).apply(mISTFT->window(),
                           [](double &x, double &y) { x *= y; });
        });

    // TODO: if normalise
    RealMatrix unnormalisedFrame =
        mFrameAndWindow(Slice(0), Slice(0, hostBufferSize));
    mOutputBuffer.pull(unnormalisedFrame);
    unnormalisedFrame.row(0).apply(unnormalisedFrame.row(1),
                                   [](double &x, double g) {
                                     if (x) {
                                       x /= g ? g : 1;
                                     }
                                   });
    if (output[0].data())
      output[0] = unnormalisedFrame.row(0);
  }

private:
  bool paramsChanged(std::size_t winSize, std::size_t hopSize,
                     std::size_t fftSize) {
    static std::size_t win, hop, fft;
    bool res = (win == winSize) && (hop == hopSize) && (fft == fftSize);

    win = winSize;
    hop = hopSize;
    fft = fftSize;

    return res;
  }

  FluidTensor<double, 2> mFrameAndWindow;
  FluidTensor<std::complex<double>, 2> mSpectrogram;
  std::unique_ptr<algorithm::STFT> mSTFT;
  std::unique_ptr<algorithm::ISTFT> mISTFT;
  BufferedProcess mBufferedProcess;
  FluidSource<double> mInputBuffer;
  FluidSink<double> mOutputBuffer;
};

} // namespace client
} // namespace fluid
