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

#include "AlgorithmUtils.hpp"
#include "FFT.hpp"
#include "FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Eigen>
#include <cmath>

namespace fluid {
namespace algorithm {

namespace impl {
// Based on https://github.com/jiixyj/libebur128/blob/master/ebur128/ebur128.c
class Interpolator
{

  struct filter
  {
    unsigned int  count; /* Number of coefficients in this subfilter */
    unsigned int* index; /* Delay index of corresponding filter coeff */
    double*       coeff; /* List of subfilter coefficients */
  };

public:
  Interpolator(index maxtaps, index maxfactor, Allocator& alloc)
      : mMaxTaps{maxtaps}, mMaxFactor{maxfactor},
        mMaxLatency{(mMaxTaps + mMaxFactor - 1) / mMaxFactor},
        mBuffer(asUnsigned(mMaxLatency), alloc), mCount(asUnsigned(mMaxFactor), alloc),
        mFilters(mMaxFactor, mMaxLatency, alloc),
        mIndex(mMaxFactor, mMaxLatency, alloc)
  {}

  void init(index taps, index factor)
  {
    assert(taps <= mMaxTaps);
    assert(factor <= mMaxFactor);
    assert(factor > 0);
    assert(taps > 0);

    mTaps = taps;
    mFactor = factor;
    mLatency = (taps + factor - 1) / factor;

    constexpr double almostZero = 1e-6;
    mHead = 0;
    std::fill(mBuffer.begin(), mBuffer.end(), 0.0);
    std::fill(mCount.begin(), mCount.end(), 0);

    mIndex.fill(0);
    mFilters.fill(0);

    for (index i = 0; i < taps; ++i)
    {
      double m = i - (taps - 1) / 2.0;
      double c = 1.0;

      if (std::abs(m) > almostZero)
      {
        c = sin(m * pi / factor) / (m * pi / factor);
      }
      c *= 0.5 * (1 - cos(twoPi * i / (taps - 1)));

      if (std::abs(c) > almostZero)
      {
        index f = i % factor;
        index t = mCount[asUnsigned(f)]++;
        mFilters(f, t) = c;
        mIndex(f, t) = i / factor;
      }
    }

    mInitialized = true;
  }

  void processFrame(FluidTensorView<const double, 1> in,
                    FluidTensorView<double, 1>       out)
  {
    assert(mInitialized);
    assert(in.size());
    assert(out.size() >= mFactor * in.size());

    auto outP = out.begin();

    for (auto& x : in)
    {

      mBuffer[asUnsigned(mHead)] = x;
      for (index i = 0; i < mFactor; ++i)
      {
        double acc = 0;
        for (index j = 0, count = mCount[asUnsigned(i)]; j < count; ++j)
        {
          index offset = mHead - mIndex(i, j);
          if (offset < 0) { offset += mLatency; }
          double c = mFilters(i, j);
          acc += mBuffer[asUnsigned(offset)] * c;
        }
        *outP = acc;
        std::advance(outP, 1);
      }
      mHead++;
      if (mHead == mLatency) { mHead = 0; }
    }
  }

private:
  bool mInitialized{false};

  index mMaxTaps;
  index mMaxFactor;
  index mMaxLatency;

  index mTaps;
  index mFactor;
  index mLatency;

  rt::vector<double>     mBuffer;
  rt::vector<index>      mCount;
  FluidTensor<double, 2> mFilters;
  FluidTensor<index, 2>  mIndex;
  index                  mHead;
};


} // namespace impl


class TruePeak
{
  static constexpr index nTaps = 49;
  static constexpr index maxFactor = 4;

public:
  TruePeak(index maxSize, Allocator& alloc)
      : mInterpolator(nTaps, maxFactor, alloc), mBuffer{maxFactor * maxSize,
                                                        alloc}
  {}

  void init(index /*size*/, double sampleRate, Allocator&)
  {
    mSampleRate = sampleRate;
    mFactor = sampleRate < (2 * 44100) ? 4 : 2;
    mInterpolator.init(nTaps, mFactor);
  }

  double processFrame(const RealVectorView& input, Allocator&)
  {
    using namespace Eigen;

    index outSize = input.size() * mFactor;
    assert(outSize <= mBuffer.size());

    if (mSampleRate >= (4 * 44100))
    {
      return _impl::asEigen<Array>(input).abs().maxCoeff();
    }
    else
    {
      auto output = mBuffer(Slice(0, outSize));
      output.fill(0);
      mInterpolator.processFrame(input, output);
      return _impl::asEigen<Array>(output).abs().maxCoeff();
    }
  }

private:
  impl::Interpolator     mInterpolator;
  double                 mSampleRate{44100.0};
  index                  mFactor{4};
  FluidTensor<double, 1> mBuffer;
};
} // namespace algorithm
} // namespace fluid
