/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
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
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <memory>

namespace fluid {
namespace client {


template <typename T>
using HostVector = FluidTensorView<T, 1>;
template <typename T>
using HostMatrix = FluidTensorView<T, 2>;

class BufferedProcess
{

public:
  BufferedProcess(index maxFramesIn, index maxFramesOut, index maxChannelsIn,
      index maxChannelsOut, index hostSize,
      Allocator& alloc = FluidDefaultAllocator())
      : mHostSize(hostSize),
        mMaxHostSize(hostSize),
        mSource(maxFramesIn, maxChannelsIn, mMaxHostSize, alloc),
        mSink(maxFramesOut, maxChannelsOut, mMaxHostSize, alloc),
        mFrameIn(asUnsigned(maxChannelsIn * maxFramesIn), alloc),
        mFrameOut(asUnsigned(maxChannelsOut * maxFramesOut), alloc)
  {}

  template <typename F>
  void process(index windowSizeIn, index windowSizeOut, index hopSize,
      FluidContext& c, F processFunc)
  {
    assert(
        windowSizeIn <= maxWindowSizeIn() && "Window in bigger than maximum");
    assert(windowSizeOut <= maxWindowSizeOut() &&
           "Window out bigger than maximum");
    for (; mFrameTime < mHostSize; mFrameTime += hopSize)
    {
      RealMatrixView windowIn{mFrameIn.data(), 0, channelsIn(), windowSizeIn};
      RealMatrixView windowOut{
          mFrameOut.data(), 0, channelsOut(), windowSizeOut};
      mSource.pull(windowIn, mFrameTime);
      processFunc(windowIn, windowOut);
      mSink.push(windowOut, mFrameTime);

      if (FluidTask* t = c.task())
        if (!t->processUpdate(
                static_cast<double>(std::min(mFrameTime + hopSize, mHostSize)),
                static_cast<double>(mHostSize)))
          break;
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  template <typename F>
  void processInput(
      index windowSize, index hopSize, FluidContext& c, F processFunc)
  {
    assert(windowSize <= maxWindowSizeIn() && "Window bigger than maximum");
    for (; mFrameTime < mHostSize; mFrameTime += hopSize)
    {
      RealMatrixView windowIn{mFrameIn.data(), 0, channelsIn(), windowSize};
      mSource.pull(windowIn, mFrameTime);
      processFunc(windowIn);

      if (FluidTask* t = c.task())
        if (!t->processUpdate(
                static_cast<double>(std::min(mFrameTime + hopSize, mHostSize)),
                static_cast<double>(mHostSize)))
          break;
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  template <typename F>
  void processOutput(
      index windowSizeOut, index hopSize, FluidContext& c, F processFunc)
  {
    assert(windowSizeOut <= maxWindowSizeOut() &&
           "Window out bigger than maximum");
    for (; mFrameTime < mHostSize; mFrameTime += hopSize)
    {
      RealMatrixView windowOut{
          mFrameOut.data(), 0, channelsOut(), windowSizeOut};
      processFunc(windowOut);
      mSink.push(windowOut, mFrameTime);

      if (FluidTask* t = c.task())
        if (!t->processUpdate(
                static_cast<double>(std::min(mFrameTime + hopSize, mHostSize)),
                static_cast<double>(mHostSize)))
          break;
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  index hostSize() const noexcept { return mHostSize; }

  void hostSize(index size) noexcept
  {
    assert(size <= mMaxHostSize);
    mHostSize = size;
    mSource.setHostBufferSize(size);
    mSink.setHostBufferSize(size);
    reset();
  }

  index maxWindowSizeIn() const noexcept { return mSource.size(); }
  index maxWindowSizeOut() const noexcept { return mSink.size(); }

  template <typename T>
  void push(HostMatrix<T> in)
  {
    mSource.push(in);
  }

  template <typename T>
  void push(const std::vector<FluidTensorView<T, 1>>& in)
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

  void reset()
  {
    mSource.reset();
    mSink.reset();
    mFrameTime = 0;
  }

private:
  index               mFrameTime = 0;
  index               mHostSize;
  index               mMaxHostSize;
  FluidSource<double> mSource;
  FluidSink<double>   mSink;
  rt::vector<double>  mFrameIn;
  rt::vector<double>  mFrameOut;
};

template <bool Normalise = true>
class STFTBufferedProcess
{

public:
  STFTBufferedProcess(FFTParams fftParams, index channelsIn, index channelsOut,
      index hostVectorSize, Allocator& alloc)
      : mBufferedProcess(fftParams.max(), fftParams.max(), channelsIn,
            channelsOut + Normalise, hostVectorSize, alloc),
        mSpectrumIn(asUnsigned(channelsIn * fftParams.maxFrameSize()), alloc),
        mSpectrumOut(asUnsigned(channelsOut * fftParams.maxFrameSize()), alloc),
        mFrameAndWindow(
            asUnsigned((Normalise + channelsOut) * fftParams.max()), alloc),
        mSTFT(fftParams.max(), fftParams.max(), fftParams.hopSize(), 0, alloc),
        mISTFT(fftParams.max(), fftParams.max(), fftParams.hopSize(), 0, alloc)
  {}


  template <typename T, typename F>
  void process(FFTParams p, const std::vector<HostVector<T>>& input,
      std::vector<HostVector<T>>& output, FluidContext& c, F&& processFunc)
  {

    if (!input[0].data()) return;
    assert(mBufferedProcess.channelsIn() == asSigned(input.size()));
    assert(
        mBufferedProcess.channelsOut() == asSigned(output.size() + Normalise));

    FFTParams fftParams = setup(p);
    index     chansIn = mBufferedProcess.channelsIn();
    index     chansOut = mBufferedProcess.channelsOut() - Normalise;

    mBufferedProcess.push(input);

    ComplexMatrixView spectrumIn{
        mSpectrumIn.data(), 0, chansIn, fftParams.frameSize()};
    ComplexMatrixView spectrumOut{
        mSpectrumOut.data(), 0, chansOut, fftParams.frameSize()};


    mBufferedProcess.process(fftParams.winSize(), fftParams.winSize(),
        fftParams.hopSize(), c,
        [this, spectrumIn, spectrumOut, &processFunc, chansIn, chansOut](
            RealMatrixView in, RealMatrixView out) {
          for (index i = 0; i < chansIn; ++i)
            mSTFT.processFrame(in.row(i), spectrumIn.row(i));
          processFunc(spectrumIn, spectrumOut(Slice(0, chansOut), Slice(0)));
          for (index i = 0; i < chansOut; ++i)
            mISTFT.processFrame(spectrumOut.row(i), out.row(i));

          if (Normalise)
          {
            out.row(chansOut) <<= mSTFT.window();
            out.row(chansOut).apply(
                mISTFT.window(), [](double& x, double& y) { x *= y; });
          }
        });

    RealMatrixView unnormalisedFrame{
        mFrameAndWindow.data(), 0, Normalise + chansOut, input[0].size()};
    //        mFrameAndWindow(Slice(0), Slice(0, input[0].size()));
    mBufferedProcess.pull(unnormalisedFrame);
    for (index i = 0; i < chansOut; ++i)
    {
      if (Normalise)
        unnormalisedFrame.row(i).apply(
            unnormalisedFrame.row(chansOut), [](double& x, double g) {
              if (x != 0) { x /= (g > 0) ? g : 1; }
            });
      if (output[asUnsigned(i)].data())
        output[asUnsigned(i)] <<= unnormalisedFrame.row(i);
    }
  }

  template <typename T, typename F>
  void processInput(FFTParams p, const std::vector<HostVector<T>>& input,
      FluidContext& c, F&& processFunc)
  {

    if (!input[0].data()) return;
    assert(mBufferedProcess.channelsIn() == asSigned(input.size()));
    index     chansIn = mBufferedProcess.channelsIn();
    FFTParams fftParams = setup(p);

    mBufferedProcess.push(input);
    ComplexMatrixView spectrumIn{
        mSpectrumIn.data(), 0, chansIn, fftParams.frameSize()};

    mBufferedProcess.processInput(fftParams.winSize(), fftParams.hopSize(), c,
        [this, spectrumIn, &processFunc, chansIn](RealMatrixView in) {
          for (index i = 0; i < chansIn; ++i)
            mSTFT.processFrame(in.row(i), spectrumIn.row(i));
          processFunc(spectrumIn);
        });
  }


  template <typename T, typename F>
  void processOutput(FFTParams p, std::vector<HostVector<T>>& output,
      FluidContext& c, F&& processFunc)
  {
    assert(
        mBufferedProcess.channelsOut() == asSigned(output.size() + Normalise));
    FFTParams fftParams = setup(p);
    index     chansOut = mBufferedProcess.channelsOut() - Normalise;

    ComplexMatrixView spectrumOut{
        mSpectrumOut.data(), 0, chansOut + Normalise, fftParams.frameSize()};

    mBufferedProcess.processOutput(fftParams.winSize(), fftParams.hopSize(), c,
        [this, spectrumOut, &processFunc, chansOut](RealMatrixView out) {
          processFunc(spectrumOut(Slice(0, chansOut), Slice(0)));
          for (index i = 0; i < chansOut; ++i)
          {
            mISTFT.processFrame(spectrumOut.row(i), out.row(i));
          }

          if (Normalise)
          {
            out.row(chansOut) <<= mSTFT.window();
            out.row(chansOut).apply(
                mISTFT.window(), [](double& x, double& y) { x *= y; });
          }
        });

    RealMatrixView unnormalisedFrame{
        mFrameAndWindow.data(), 0, Normalise + chansOut, output[0].size()};
    mBufferedProcess.pull(unnormalisedFrame);
    for (index i = 0; i < chansOut; ++i)
    {
      if (Normalise)
        unnormalisedFrame.row(i).apply(
            unnormalisedFrame.row(chansOut), [](double& x, double g) {
              if (x != 0) { x /= (g > 0) ? g : 1; }
            });
      if (output[asUnsigned(i)].data())
        output[asUnsigned(i)] <<= unnormalisedFrame.row(i);
    }
  }

  void reset() { mBufferedProcess.reset(); }

private:
  FFTParams setup(FFTParams fftParams)
  {

    bool newParams = mTrackValues.changed(
        fftParams.winSize(), fftParams.hopSize(), fftParams.fftSize());

    if (newParams)
    {
      mSTFT.resize(
          fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());
      mISTFT.resize(
          fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());
    }

    return fftParams;
  }

  ParameterTrackChanges<index, index, index> mTrackValues;
  BufferedProcess                            mBufferedProcess;
  rt::vector<std::complex<double>>           mSpectrumIn;
  rt::vector<std::complex<double>>           mSpectrumOut;
  rt::vector<double>                         mFrameAndWindow;
  algorithm::STFT                            mSTFT;
  algorithm::ISTFT                           mISTFT;
};

} // namespace client
} // namespace fluid
