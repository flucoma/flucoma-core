/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/FluidContext.hpp"
#include "../common/FluidSink.hpp"
#include "../common/FluidSource.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <memory>

namespace fluid {
namespace client {


class BufferedProcess
{

  template <typename T>
  using HostVector = FluidTensorView<T, 1>;
  template <typename T>
  using HostMatrix = FluidTensorView<T, 2>;

public:
  template <typename F>
  void process(index windowSizeIn, index windowSizeOut,
               index hopSize, FluidContext& c, bool reset, F processFunc)
  {
    assert(windowSizeIn <= maxWindowSizeIn() &&
           "Window in bigger than maximum");
    assert(windowSizeOut <= maxWindowSizeOut() &&
           "Window out bigger than maximum");
    if (reset) mFrameTime = 0;
    for (; mFrameTime < mHostSize; mFrameTime += hopSize)
    {
      RealMatrixView windowIn = mFrameIn(Slice(0), Slice(0, windowSizeIn));
      RealMatrixView windowOut = mFrameOut(Slice(0), Slice(0, windowSizeOut));
      mSource.pull(windowIn, mFrameTime);
      processFunc(windowIn, windowOut);
      mSink.push(windowOut, mFrameTime);

      if (FluidTask* t = c.task())
        if (!t->processUpdate(static_cast<double>(std::min(mFrameTime + hopSize, mHostSize)),
                              static_cast<double>(mHostSize)))
          break;
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  template <typename F>
  void processInput(index windowSize, index hopSize,
                    FluidContext& c, bool reset, F processFunc)
  {
    assert(windowSize <= maxWindowSizeIn() && "Window bigger than maximum");
    if (reset) mFrameTime = 0;
    for (; mFrameTime < mHostSize; mFrameTime += hopSize)
    {
      RealMatrixView windowIn = mFrameIn(Slice(0), Slice(0, windowSize));
      mSource.pull(windowIn, mFrameTime);
      processFunc(windowIn);

      if (FluidTask* t = c.task())
        if (!t->processUpdate(static_cast<double>(std::min(mFrameTime + hopSize, mHostSize)),
                              static_cast<double>(mHostSize)))
          break;
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  index hostSize() const noexcept { return mHostSize; }
  void        hostSize(index size) noexcept
  {
    mHostSize = size;
    mSource.setHostBufferSize(size);
    mSink.setHostBufferSize(size);
    mSource.reset();
    mSink.reset();
  }

  index maxWindowSizeIn() const noexcept { return mFrameIn.cols(); }
  index maxWindowSizeOut() const noexcept { return mFrameOut.cols(); }

  void maxSize(index framesIn, index framesOut,
               index channelsIn, index channelsOut)
  {
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

  template <typename T>
  void push(HostMatrix<T> in)
  {
    mSource.push(in);
  }
  template <typename T>
  void pull(HostMatrix<T> out)
  {
    mSink.pull(out);
  }

  index channelsIn() const noexcept { return mSource.channels(); }
  index channelsOut() const noexcept { return mSink.channels(); }

private:
  index         mFrameTime = 0;
  index         mHostSize;
  RealMatrix          mFrameIn;
  RealMatrix          mFrameOut;
  FluidSource<double> mSource;
  FluidSink<double>   mSink;
};

template <typename Params, typename U, index FFTParamsIndex,
          bool Normalise = true>
class STFTBufferedProcess
{
  using HostVector = FluidTensorView<U, 1>;
  using HostMatrix = FluidTensorView<U, 2>;

public:
  STFTBufferedProcess(index maxFFTSize, index channelsIn, index channelsOut)
  {
    mBufferedProcess.maxSize(maxFFTSize, maxFFTSize, channelsIn,
                             channelsOut + Normalise);
  }


  template <typename F>
  void process(Params& p, std::vector<HostVector>& input,
               std::vector<HostVector>& output, FluidContext& c, bool reset,
               F&& processFunc)
  {

    if (!input[0].data()) return;
    assert(mBufferedProcess.channelsIn() == asSigned(input.size()));
    assert(mBufferedProcess.channelsOut() == asSigned(output.size() + Normalise));

    FFTParams fftParams = setup(p, input);
    index    chansIn = mBufferedProcess.channelsIn();
    index    chansOut = mBufferedProcess.channelsOut() - Normalise;
    mBufferedProcess.process(
        fftParams.winSize(), fftParams.winSize(), fftParams.hopSize(), c, reset,
        [this, &processFunc, chansIn, chansOut](RealMatrixView in,
                                                RealMatrixView out) {
          for (index i = 0; i < chansIn; ++i)
            mSTFT->processFrame(in.row(i), mSpectrumIn.row(i));
          processFunc(mSpectrumIn, mSpectrumOut(Slice(0, chansOut), Slice(0)));
          for (index i = 0; i < chansOut; ++i)
            mISTFT->processFrame(mSpectrumOut.row(i), out.row(i));
          if (Normalise)
          {
            out.row(chansOut) = mSTFT->window();
            out.row(chansOut).apply(mISTFT->window(),
                                    [](double& x, double& y) { x *= y; });
          }
        });

    if (Normalise)
    {
      RealMatrixView unnormalisedFrame =
          mFrameAndWindow(Slice(0), Slice(0, input[0].size()));
      mBufferedProcess.pull(unnormalisedFrame);
      for (index i = 0; i < chansOut; ++i)
      {
        unnormalisedFrame.row(i).apply(unnormalisedFrame.row(chansOut),
                                       [](double& x, double g) {
                                         if (x > 0) { x /= (g>0) ? g : 1; }
                                       });
        if (output[asUnsigned(i)].data()) output[asUnsigned(i)] = unnormalisedFrame.row(i);
      }
    }
  }

  template <typename F>
  void processInput(Params& p, std::vector<HostVector>& input, FluidContext& c,
                    bool reset, F&& processFunc)
  {

    if (!input[0].data()) return;
    assert(mBufferedProcess.channelsIn() == asSigned(input.size()));
    index    chansIn = mBufferedProcess.channelsIn();
    FFTParams fftParams = setup(p, input);

    mBufferedProcess.processInput(
        fftParams.winSize(), fftParams.hopSize(), c, reset,
        [this, &processFunc, chansIn](RealMatrixView in) {
          for (index i = 0; i < chansIn; ++i)
            mSTFT->processFrame(in.row(i), mSpectrumIn.row(i));
          processFunc(mSpectrumIn);
        });
  }

private:
  FFTParams setup(Params& p, std::vector<HostVector>& input)
  {
    FFTParams fftParams = p.template get<FFTParamsIndex>();
    bool      newParams = mTrackValues.changed(
        fftParams.winSize(), fftParams.hopSize(), fftParams.fftSize());
    index hostBufferSize = input[0].size();
    if (mTrackHostVS.changed(hostBufferSize))
      mBufferedProcess.hostSize(hostBufferSize);

    if (!mSTFT.get() || newParams)
      mSTFT.reset(new algorithm::STFT(fftParams.winSize(), fftParams.fftSize(),
                                      fftParams.hopSize()));
    if (!mISTFT.get() || newParams)
      mISTFT.reset(new algorithm::ISTFT(
          fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize()));

    index chansIn = mBufferedProcess.channelsIn();
    index chansOut = mBufferedProcess.channelsOut();

    if (fftParams.frameSize() != mSpectrumIn.cols())
      mSpectrumIn.resize(chansIn, fftParams.frameSize());

    if (fftParams.frameSize() != mSpectrumOut.cols())
      mSpectrumOut.resize(chansOut, fftParams.frameSize());

    if ((std::max(mBufferedProcess.maxWindowSizeIn(),
                              hostBufferSize) > mFrameAndWindow.cols()) && Normalise)
      mFrameAndWindow.resize(
          chansOut,
          std::max(mBufferedProcess.maxWindowSizeIn(), hostBufferSize));

    mBufferedProcess.push(HostMatrix(input[0]));
    return fftParams;
  }

  ParameterTrackChanges<index, index, index> mTrackValues;
  ParameterTrackChanges<index>                 mTrackHostVS;
  RealMatrix                                    mFrameAndWindow;
  ComplexMatrix                                 mSpectrumIn;
  ComplexMatrix                                 mSpectrumOut;
  std::unique_ptr<algorithm::STFT>              mSTFT;
  std::unique_ptr<algorithm::ISTFT>             mISTFT;
  BufferedProcess                               mBufferedProcess;
};

} // namespace client
} // namespace fluid
