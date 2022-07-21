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
      : mFFT(maxSize, alloc), mWindowStorage(asUnsigned(maxSize), alloc),
        mPrevFrameStorage(asUnsigned(maxSize / 2 + 1), 0, alloc),
        mPrevPrevFrameStorage(asUnsigned(maxSize / 2 + 1), 0, alloc),
        mFilter(maxFilterSize, alloc)
  {}

  void init(index windowSize, index fftSize, index filterSize)
  {
    assert(windowSize <= asSigned(mWindowStorage.size()));
    makeWindow(windowSize);

    std::fill(mPrevFrameStorage.begin(), mPrevFrameStorage.end(), 0);
    std::fill(mPrevPrevFrameStorage.begin(), mPrevPrevFrameStorage.end(), 0);
    mFilter.init(filterSize);
    mFFT.resize(fftSize);
    mDebounceCount = 1;
    mPrevFuncVal = 0;
    mInitialized = true;
  }

  void makeWindow(index windowSize)
  {
    std::fill(mWindowStorage.begin(), mWindowStorage.end(), 0);
    Eigen::Map<Eigen::ArrayXd> window(mWindowStorage.data(), windowSize);
    WindowFuncs::map()[mWindowType](windowSize, window);
    mWindowSize = windowSize;
  }

  /// input window isn't necessarily a single framre because it should encompass
  /// `frameDelta`'s worth of history
  double processFrame(RealVectorView input, index function, index filterSize,
                      index frameDelta, Allocator& alloc)
  {
    assert(mInitialized);
    FluidEigenMap<Eigen::Array> in = _impl::asEigen<Eigen::Array>(input);
    double                      funcVal = 0;
    double                      filteredFuncVal = 0;

    index frameSize = mWindowSize / 2 + 1;

    if (filterSize >= 3 &&
        (!mFilter.initialized() || filterSize != mFilter.size()))
      mFilter.init(filterSize);

    rt::vector<std::complex<double>> frameBuffer(asUnsigned(frameSize), alloc);
    Eigen::Map<ArrayXcd>             frame(frameBuffer.data(), frameSize);
    Eigen::Map<Eigen::ArrayXd>       window(mWindowStorage.data(), mWindowSize);
    frame = mFFT.process(in.col(0).segment(0, mWindowSize) * window);

    Eigen::Map<ArrayXcd> prevFrame(mPrevFrameStorage.data(), frameSize);
    Eigen::Map<ArrayXcd> prevPrevFrame(mPrevPrevFrameStorage.data(), frameSize);

    auto odf = static_cast<OnsetDetectionFuncs::ODF>(function);
    if (function > 1 && function < 5 && frameDelta != 0)
    {
      rt::vector<std::complex<double>> frameBuffer2(asUnsigned(frameSize),
                                                    alloc);
      Eigen::Map<ArrayXcd>             frame2(frameBuffer2.data(), frameSize);
      frame2 =
          mFFT.process(in.col(0).segment(frameDelta, mWindowSize) * window);
      funcVal = OnsetDetectionFuncs::map()[odf](frame2, frame, frame);
    }
    else
    {
      funcVal =
          OnsetDetectionFuncs::map()[odf](frame, prevFrame, prevPrevFrame);
    }
    if (filterSize >= 3)
      filteredFuncVal = funcVal - mFilter.processSample(funcVal);
    else
      filteredFuncVal = funcVal - mPrevFuncVal;

    prevPrevFrame = prevFrame;
    prevFrame = frame;

    return filteredFuncVal;
  }

private:
  FFT                              mFFT;
  rt::vector<double>               mWindowStorage;
  ArrayXd                          mWindow;
  index                            mWindowSize{1024};
  index                            mDebounceCount{1};
  rt::vector<std::complex<double>> mPrevFrameStorage;
  rt::vector<std::complex<double>> mPrevPrevFrameStorage;
  double                           mPrevFuncVal{0.0};
  WindowTypes                      mWindowType{WindowTypes::kHann};
  MedianFilter                     mFilter;
  bool                             mInitialized{false};
};

} // namespace algorithm
} // namespace fluid
