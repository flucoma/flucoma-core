#pragma once

#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>

#include <algorithms/public/STFT.hpp>
#include <clients/common/FluidSink.hpp>
#include <clients/common/FluidSource.hpp>
#include <clients/common/DeriveSTFTParams.hpp>
#include <clients/common/ParameterTrackChanges.hpp>

namespace fluid {
namespace client {

template<typename T>
using HostVector = FluidTensorView<T,1>;

template<typename T>
using HostMatrix = FluidTensorView<T,2>;

class BufferedProcess {
public:
  template <typename F>
  void process(std::size_t windowSize, std::size_t hopSize, F processFunc) {
    assert(windowSize <= maxWindowSize() && "Window bigger than maximum");
    for (; mFrameTime < mHostSize; mFrameTime += hopSize) {
      RealMatrixView windowIn  = mFrameIn(Slice(0), Slice(0, windowSize));
      RealMatrixView windowOut = mFrameOut(Slice(0), Slice(0, windowSize));
      
      mSource.pull(windowIn, mFrameTime);
      processFunc(windowIn, windowOut);
      mSink.push(windowOut, mFrameTime);
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  std::size_t hostSize() const noexcept { return mHostSize; }
  void hostSize(std::size_t size) noexcept
  {
    mHostSize = size;
    mSource.setHostBufferSize(size);
    mSink.setHostBufferSize(size);
  }
  
  std::size_t maxWindowSize() const noexcept { return mFrameIn.cols(); }
  void maxSize(std::size_t frames, std::size_t channelsIn,
               std::size_t channelsOut) {
    mSource.setSize(frames);
    mSource.reset(channelsIn);
    mSink.setSize(frames);
    mSink.reset(channelsOut);
    
    if (channelsIn > mFrameIn.rows() || frames > mFrameIn.cols())
      mFrameIn.resize(channelsIn, frames);
    if (channelsOut > mFrameOut.rows() || frames > mFrameOut.cols())
      mFrameOut.resize(channelsOut, frames);
  }

  template<typename T> void push(HostMatrix<T> in) { mSource.push(in);}
  template<typename T> void pull(HostMatrix<T> out){ mSink.pull(out); }

private:
  std::size_t mFrameTime = 0 ;
  std::size_t mHostSize;
  RealMatrix mFrameIn;
  RealMatrix mFrameOut;
  FluidSource<double> mSource;
  FluidSink<double> mSink;
};

template <typename T, typename U, typename Client, size_t maxWinParam,
          size_t winParam, size_t hopParam, size_t FFTParam, bool Normalise, bool Invert = true>
class STFTBufferedProcess {
  using HostVector = HostVector<U>;

public:
  template <typename F>
  void process(Client &x, std::vector<HostVector> &input,
               std::vector<HostVector> &output, F &&processFunc) {
    size_t winSize, hopSize, fftSize;
    
    std::tie(winSize, hopSize, fftSize) = impl::deriveSTFTParams<winParam, hopParam, FFTParam>(x); 
  
    bool newParams = mTrackValues.changed(winSize, hopSize, fftSize);

    if (!mSTFT.get() || newParams)
      mSTFT.reset(new algorithm::STFT(winSize, fftSize, hopSize));

    if (Invert && !mISTFT.get() || newParams)
      mISTFT.reset(new algorithm::ISTFT(winSize, fftSize, hopSize));

    if (!input[0].data())
      return; // if there's not actually an audio input, no point continuing

    std::size_t chansIn = x.audioChannelsIn();
    std::size_t chansOut = x.audioChannelsOut();

    assert(chansIn == input.size());
    if(Invert) assert(chansOut == output.size());

    size_t hostBufferSize = input[0].size();
    mBufferedProcess.hostSize(hostBufferSize); // safe assumption?

    std::size_t maxWin = x.template get<maxWinParam>();
    mBufferedProcess.maxSize(maxWin, chansIn, chansOut + 1);

    if (std::max(maxWin,hostBufferSize) > mFrameAndWindow.cols())
      mFrameAndWindow.resize(chansOut + 1, std::max(maxWin,hostBufferSize));
    
    if ((fftSize / 2 + 1) != mSpectrumIn.cols())
    {
      mSpectrumIn.resize(chansIn, (fftSize / 2 + 1));
    }
    
    if ((fftSize / 2 + 1) != mSpectrumOut.cols())
    {
      mSpectrumOut.resize(chansOut, (fftSize / 2 + 1));
    }

    mBufferedProcess.push(HostMatrix<U>(input[0]));

    mBufferedProcess.process(
        winSize, hopSize,
        [this, &processFunc, chansIn, chansOut](RealMatrixView in, RealMatrixView out) {
          for(int i = 0; i < chansIn; ++i)
            mSTFT->processFrame(in.row(i), mSpectrumIn.row(i));
          processFunc(mSpectrumIn, mSpectrumOut);
          if(Invert)
          {
            for(int i = 0; i < chansOut; ++i)
              mISTFT->processFrame(mSpectrumOut.row(i), out.row(i));
            out.row(chansOut) = mSTFT->window();
            out.row(chansOut).apply(mISTFT->window(),[](double &x, double &y) { x *= y; });
          }
        });

    if(Invert && Normalise)
    {
      RealMatrixView unnormalisedFrame = mFrameAndWindow(Slice(0), Slice(0, hostBufferSize));
      mBufferedProcess.pull(unnormalisedFrame);
      for(int i = 0; i < chansOut; ++i)
      {
        unnormalisedFrame.row(i).apply(unnormalisedFrame.row(chansOut),[](double &x, double g) {
                                         if (x) {  x /= g ? g : 1; }
                                       });
        if (output[i].data())  output[i] = unnormalisedFrame.row(i);
      }
    }
  }

private:
  ParameterTrackChanges<size_t, size_t, size_t> mTrackValues;
  RealMatrix mFrameAndWindow;
  ComplexMatrix mSpectrumIn;
  ComplexMatrix mSpectrumOut;
  std::unique_ptr<algorithm::STFT> mSTFT;
  std::unique_ptr<algorithm::ISTFT> mISTFT;
  BufferedProcess mBufferedProcess;
  size_t mWinSize{0};
  size_t mHopSize{0};
  size_t mFFTSize{0};
};

} // namespace client
} // namespace fluid
