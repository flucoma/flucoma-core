#pragma once

#include "../common/FluidSink.hpp"
#include "../common/FluidSource.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"

#include <memory>

namespace fluid {
namespace client {

template<typename T>
using HostVector = FluidTensorView<T,1>;

template<typename T>
using HostMatrix = FluidTensorView<T,2>;

class BufferedProcess {
public:
  template <typename F>
  void process(std::size_t windowSizeIn, std::size_t windowSizeOut, std::size_t hopSize, bool reset, F processFunc) {
    assert(windowSizeIn <= maxWindowSizeIn() && "Window in bigger than maximum");
    assert(windowSizeOut <= maxWindowSizeOut() && "Window out bigger than maximum");
    if(reset) mFrameTime = 0;
    for (; mFrameTime < mHostSize; mFrameTime += hopSize) {
      RealMatrixView windowIn  = mFrameIn(Slice(0), Slice(0, windowSizeIn));
      RealMatrixView windowOut = mFrameOut(Slice(0), Slice(0, windowSizeOut));
      mSource.pull(windowIn, mFrameTime);
      processFunc(windowIn, windowOut);
      mSink.push(windowOut, mFrameTime);
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }
  
  template <typename F>
  void processInput(std::size_t windowSize, std::size_t hopSize, bool reset, F processFunc) {
    assert(windowSize <= maxWindowSizeIn() && "Window bigger than maximum");
    if(reset) mFrameTime = 0;
    for (; mFrameTime < mHostSize; mFrameTime += hopSize) {
      RealMatrixView windowIn  = mFrameIn(Slice(0), Slice(0, windowSize));
      mSource.pull(windowIn, mFrameTime);
      processFunc(windowIn);
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  std::size_t hostSize() const noexcept { return mHostSize; }
  void hostSize(std::size_t size) noexcept
  {
    mHostSize = size;
    mSource.setHostBufferSize(size);
    mSink.setHostBufferSize(size);
    mSource.reset();
    mSink.reset();
  }
  
  std::size_t maxWindowSizeIn() const noexcept { return mFrameIn.cols(); }
  std::size_t maxWindowSizeOut() const noexcept { return mFrameOut.cols(); }
    
  void maxSize(std::size_t framesIn, std::size_t framesOut, std::size_t channelsIn,
               std::size_t channelsOut) {
    mSource.setSize(framesIn);
    mSource.reset(channelsIn);
    mSink.setSize(framesOut);
    mSink.reset(channelsOut);

    if (channelsIn > mFrameIn.rows() || framesIn > mFrameIn.cols())
      mFrameIn.resize(channelsIn, framesIn);
    if (channelsOut > mFrameOut.rows() || framesOut > mFrameOut.cols())
      mFrameOut.resize(channelsOut, framesOut);
      
    mFrameTime = 0;
  }

  template<typename T> void push(HostMatrix<T> in) {
    mSource.push(in);
  }
  template<typename T> void pull(HostMatrix<T> out){ mSink.pull(out); }
  
  size_t channelsIn()  const noexcept { return mSource.channels(); }
  size_t channelsOut() const noexcept { return mSink.channels(); }
private:
  std::size_t mFrameTime = 0 ;
  std::size_t mHostSize;
  RealMatrix mFrameIn;
  RealMatrix mFrameOut;
  FluidSource<double> mSource;
  FluidSink<double> mSink;
};

template <typename Params, typename U, size_t FFTParamsIndex, bool Normalise=true>
class  STFTBufferedProcess {
  using HostVector = HostVector<U>;

public:

  STFTBufferedProcess(size_t maxFFTSize, size_t channelsIn, size_t channelsOut){
    mBufferedProcess.maxSize(maxFFTSize, maxFFTSize, channelsIn, channelsOut + Normalise);
  }
  
  
  template <typename F>
  void process(Params &p, std::vector<HostVector> &input,
               std::vector<HostVector> &output, bool reset, F &&processFunc) {
   
    if (!input[0].data()) return;
    assert(mBufferedProcess.channelsIn() == input.size());
    assert(mBufferedProcess.channelsOut() == output.size() + Normalise);
    
    FFTParams fftParams = setup(p, input);
    size_t chansIn = mBufferedProcess.channelsIn() ;
    size_t chansOut = mBufferedProcess.channelsOut() - Normalise ;
    mBufferedProcess.process(fftParams.winSize(), fftParams.winSize(), fftParams.hopSize(), reset,
        [this, &processFunc, chansIn, chansOut](RealMatrixView in, RealMatrixView out) {

          for(int i = 0; i < chansIn; ++i)
            mSTFT->processFrame(in.row(i), mSpectrumIn.row(i));
          processFunc(mSpectrumIn, mSpectrumOut(Slice(0,chansOut),Slice(0)));
          for(int i = 0; i < chansOut; ++i)
            mISTFT->processFrame(mSpectrumOut.row(i), out.row(i));
          if(Normalise)
          {
            out.row(chansOut) = mSTFT->window();
            out.row(chansOut).apply(mISTFT->window(),[](double &x, double &y) { x *= y; });
          }
        });

    if(Normalise)
    {
      RealMatrixView unnormalisedFrame = mFrameAndWindow(Slice(0), Slice(0, input[0].size()));
      mBufferedProcess.pull(unnormalisedFrame);
      for(int i = 0; i < chansOut; ++i)
      {
        unnormalisedFrame.row(i).apply(unnormalisedFrame.row(chansOut),[](double &x, double g) {
                                         if (x) { x /= g ? g : 1; }
                                       });
        if (output[i].data()) output[i] = unnormalisedFrame.row(i);
      }
    }
  }
  
  template <typename F>
  void processInput(Params &p, std::vector<HostVector> &input,bool reset,F &&processFunc) {
   
    if (!input[0].data()) return;
    assert(mBufferedProcess.channelsIn() == input.size());
    size_t chansIn = mBufferedProcess.channelsIn();
    FFTParams fftParams = setup(p, input);

    mBufferedProcess.processInput(fftParams.winSize(), fftParams.hopSize(), reset,
        [this, &processFunc, chansIn](RealMatrixView in) {
          for(int i = 0; i < chansIn; ++i) mSTFT->processFrame(in.row(i), mSpectrumIn.row(i));
          processFunc(mSpectrumIn);
        });
  }

private:

  FFTParams setup(Params &p, std::vector<HostVector> &input)
  {
    FFTParams fftParams = p.template get<FFTParamsIndex>();
    bool newParams = mTrackValues.changed(fftParams.winSize(), fftParams.hopSize(), fftParams.fftSize());
    size_t hostBufferSize = input[0].size();
    if(mTrackHostVS.changed(hostBufferSize))
      mBufferedProcess.hostSize(hostBufferSize);

    if (!mSTFT.get() || newParams)
      mSTFT.reset(new algorithm::STFT(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize()));
    if (!mISTFT.get() || newParams)
      mISTFT.reset(new algorithm::ISTFT(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize()));

    std::size_t chansIn = mBufferedProcess.channelsIn();
    std::size_t chansOut = mBufferedProcess.channelsOut();

    if (fftParams.frameSize() != mSpectrumIn.cols())
      mSpectrumIn.resize(chansIn, fftParams.frameSize());
    
    if (fftParams.frameSize() != mSpectrumOut.cols())
      mSpectrumOut.resize(chansOut, fftParams.frameSize());
    
    if (Normalise && std::max(mBufferedProcess.maxWindowSizeIn(), hostBufferSize) > mFrameAndWindow.cols())
      mFrameAndWindow.resize(chansOut, std::max(mBufferedProcess.maxWindowSizeIn(), hostBufferSize));
    
    mBufferedProcess.push(HostMatrix<U>(input[0]));
    return fftParams;
  }

  ParameterTrackChanges<size_t, size_t, size_t> mTrackValues;
  ParameterTrackChanges<size_t> mTrackHostVS;
  RealMatrix mFrameAndWindow;
  ComplexMatrix mSpectrumIn;
  ComplexMatrix mSpectrumOut;
  std::unique_ptr<algorithm::STFT> mSTFT;
  std::unique_ptr<algorithm::ISTFT> mISTFT;
  BufferedProcess mBufferedProcess;
};

} // namespace client
} // namespace fluid
