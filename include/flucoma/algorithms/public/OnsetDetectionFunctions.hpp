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

#include "WindowFuncs.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/MedianFilter.hpp"
#include "../util/OnsetDetectionFuncs.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>

namespace fluid {
namespace algorithm {

class OnsetDetectionFunctions
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXcd = Eigen::ArrayXcd;
  using WindowTypes = WindowFuncs::WindowTypes;

public:
  OnsetDetectionFunctions(index maxSize, index maxFilterSize, Allocator& alloc)
      : mFFT(maxSize, alloc), mWindow(maxSize, alloc),
        mPrevFrame(maxSize / 2 + 1, alloc),
        mPrevPrevFrame(maxSize / 2 + 1, alloc), mFilter(maxFilterSize, alloc)
  {}

  void init(index windowSize, index fftSize, index filterSize)
  {
    assert(windowSize <= mWindow.size());
    makeWindow(windowSize);
    mPrevFrame.setZero();
    mPrevPrevFrame.setZero();
    mFilter.init(std::max<index>(filterSize, 3));
    mFFT.resize(fftSize);
    assert(fftSize <= mWindow.size());
    mFFTSize = fftSize;
    mDebounceCount = 1;
    mPrevFuncVal = 0;
    mInitialized = true;
  }

  void makeWindow(index windowSize)
  {
    mWindow.setZero();
    WindowFuncs::map()[mWindowType](windowSize, mWindow.head(windowSize));
    mWindowSize = windowSize;
  }

  index nextPower2(index n)
  {
    return static_cast<index>(std::pow(2, std::ceil(std::log2(n))));
  }

  /// input window isn't necessarily a single frame because it should encompass
  /// `frameDelta`'s worth of history
  double processFrame(RealVectorView input, index function, index filterSize,
                      index frameDelta, Allocator& alloc)
  {
    assert(mInitialized);
    double funcVal = 0;
    double filteredFuncVal = 0;
    index frameSize = nextPower2(mFFTSize) / 2 + 1;

    if (filterSize >= 3 &&
        (!mFilter.initialized() || filterSize != mFilter.size()))
      mFilter.init(filterSize);

    ScopedEigenMap<ArrayXd> in(mFFTSize, alloc);
    in.head(mWindowSize) =
        _impl::asEigen<Eigen::Array>(input).col(0).head(mWindowSize) *
        mWindow.head(mWindowSize);

    ScopedEigenMap<ArrayXcd> frame(frameSize, alloc);
    frame = mFFT.process(in);

    if (function > 1 && function < 5 && frameDelta != 0)
    {
      ScopedEigenMap<ArrayXcd> frame2(frameSize, alloc);
      ScopedEigenMap<ArrayXd>  in2(mFFTSize, alloc);
      in2.head(mWindowSize) = _impl::asEigen<Eigen::Array>(input)(
                                  Eigen::seqN(frameDelta, mWindowSize), 0) *
                              mWindow.head(mWindowSize);
      frame2 = mFFT.process(in2);
      funcVal = OnsetDetectionFuncs::map(function)(frame2, frame, frame, alloc);
    }
    else
    {
      funcVal = OnsetDetectionFuncs::map(function)(
          frame, mPrevFrame.head(frameSize), mPrevPrevFrame.head(frameSize),
          alloc);
    }
    if (filterSize >= 3)
      filteredFuncVal = funcVal - mFilter.processSample(funcVal);
    else
      filteredFuncVal = funcVal - mPrevFuncVal;

    mPrevPrevFrame.head(frameSize) = mPrevFrame.head(frameSize);
    mPrevFrame.head(frameSize) = frame;
    return filteredFuncVal;
  }

private:
  FFT                      mFFT;
  ScopedEigenMap<ArrayXd>  mWindow;
  index                    mWindowSize{1024};
  index                    mFFTSize;
  index                    mDebounceCount{1};
  ScopedEigenMap<ArrayXcd> mPrevFrame;
  ScopedEigenMap<ArrayXcd> mPrevPrevFrame;
  double                   mPrevFuncVal{0.0};
  WindowTypes              mWindowType{WindowTypes::kHann};
  MedianFilter             mFilter;
  bool                     mInitialized{false};
};

} // namespace algorithm
} // namespace fluid
