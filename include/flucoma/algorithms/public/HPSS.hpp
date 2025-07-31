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

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/MedianFilter.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class HPSS
{
public:
  
  template<typename T>
  using Container = rt::vector<T>;
  
  enum HPSSMode { kClassic, kCoupled, kAdvanced };

  HPSS(index maxFFTSize, index maxHSize,Allocator& alloc)
      : mMaxBins(maxFFTSize / 2 + 1),mMaxHSize(maxHSize),
        mHBuf(asUnsigned(mMaxBins *  maxHSize), 0, alloc),
        mVBuf(asUnsigned(mMaxBins *  maxHSize), 0, alloc),
        mFrameBuf(asUnsigned(mMaxBins * maxHSize), 0,  alloc),
        mPaddedBuf(asUnsigned(mMaxBins * 3), 0, alloc),
        mHFilters(asUnsigned(mMaxBins),MedianFilter(mMaxHSize, alloc), alloc),
        mVFilter(mMaxBins, alloc),
        mHarmMaskBuf(asUnsigned(mMaxBins),alloc),
        mPercMaskBuf(asUnsigned(mMaxBins),alloc),
        mResMaskBuf(asUnsigned(mMaxBins),alloc),
        mMaskNormBuf(asUnsigned(mMaxBins),alloc),
        mMaskThreshBuf(asUnsigned(mMaxBins),alloc)
  {}

  void init(index nBins, index hSize)
  {
    using namespace Eigen;
    assert(hSize % 2);
    assert(nBins <= mMaxBins);
    assert(hSize <= mMaxHSize);

    ArrayXXMap v(mVBuf.data(),nBins, hSize);
    ArrayXXMap h(mHBuf.data(),nBins,hSize);
    ArrayXXcMap buf(mFrameBuf.data(),nBins,hSize);
    h.setZero();
    v.setZero();
    buf.setZero();

    for (index i = 0; i < nBins; i++) { mHFilters[asUnsigned(i)].init(hSize); }
    mInitialized = true;
  }

  void processFrame(const ComplexVectorView in, ComplexMatrixView out,
                    index vSize, index hSize, index mode, double hThresholdX1,
                    double hThresholdY1, double hThresholdX2,
                    double hThresholdY2, double pThresholdX1,
                    double pThresholdY1, double pThresholdX2,
                    double pThresholdY2)
  {
    using namespace Eigen;
    assert(mInitialized);
    assert(vSize <= in.size());
    assert(in.size() <= mMaxBins);

    index    h2 = (hSize - 1) / 2;
    index    v2 = (vSize - 1) / 2;
    index    nBins = in.size();

    ArrayXXMap v(mVBuf.data(),nBins, hSize);
    ArrayXXMap h(mHBuf.data(),nBins,hSize);
    ArrayXXcMap buf(mFrameBuf.data(),nBins,hSize);
    
    v.block(0, 0, nBins, hSize - 1) = v.block(0, 1, nBins, hSize - 1);
    h.block(0, 0, nBins, hSize - 1) = h.block(0, 1, nBins, hSize - 1);
    buf.block(0, 0, nBins, hSize - 1) = buf.block(0, 1, nBins, hSize - 1);
    
    ArrayXMap padded(mPaddedBuf.data(),2 * vSize + nBins);
    padded.setZero();

    padded.segment(v2, nBins) = _impl::asEigen<Array>(in).abs().real();
    mVFilter.init(vSize);
    for (index i = 0; i < padded.size(); i++)
    {
      padded(i) = mVFilter.processSample(padded(i));
    }

    v.block(0, hSize - 1, nBins, 1) = padded.segment(v2 * 3, nBins);
    buf.block(0, hSize - 1, nBins, 1) = _impl::asEigen<Array>(in);
    h.block(0, h2 + 1, nBins, 1) = ArrayXd::NullaryExpr(nBins,[&in,this](int i){
       return mHFilters[asUnsigned(i)].processSample(std::abs(in(i)));
    });
    
    ArrayXMap harmonicMask(mHarmMaskBuf.data(),nBins);
    ArrayXMap percussiveMask(mPercMaskBuf.data(),nBins);
    ArrayXMap residualMask(mResMaskBuf.data(),nBins);
    residualMask =
        mode == kAdvanced ? ArrayXd::Ones(nBins) : ArrayXd::Zero(nBins);
    switch (mode)
    {
    case kClassic: {
      ArrayXMap mult(mMaskNormBuf.data(),nBins);
      mult  = (1.0 / (h.col(0) + v.col(0)).max(epsilon));
      harmonicMask = (h.col(0) * mult);
      percussiveMask = (v.col(0) * mult);
      break;
    }
    case kCoupled: {
      harmonicMask = ((h.col(0) / v.col(0)) >
                      makeThreshold(nBins, hThresholdX1, hThresholdY1,
                                    hThresholdX2, hThresholdY2))
                         .cast<double>();
      percussiveMask = 1 - harmonicMask;
      break;
    }
    case kAdvanced: {
      harmonicMask = ((h.col(0) / v.col(0)) >
                      makeThreshold(nBins, hThresholdX1, hThresholdY1,
                                    hThresholdX2, hThresholdY2))
                         .cast<double>();
      percussiveMask = ((v.col(0) / h.col(0)) >
                        makeThreshold(nBins, pThresholdX1, pThresholdY1,
                                      pThresholdX2, pThresholdY2))
                           .cast<double>();
  
      residualMask = residualMask * (1 - harmonicMask);
      residualMask = residualMask * (1 - percussiveMask);
      ArrayXMap maskNorm(mMaskNormBuf.data(),nBins);      
      maskNorm =
          (1. / (harmonicMask + percussiveMask + residualMask)).max(epsilon);
      harmonicMask = harmonicMask * maskNorm;
      percussiveMask = percussiveMask * maskNorm;
      residualMask = residualMask * maskNorm;
      break;
    }
    }
    _impl::asEigen<Array>(out).col(0) = buf.col(0) * harmonicMask.min(1.0);
    _impl::asEigen<Array>(out).col(1) = buf.col(0) * percussiveMask.min(1.0);
    _impl::asEigen<Array>(out).col(2) = buf.col(0) * residualMask.min(1.0);
  }
  
  bool initialized() const { return mInitialized; }

private:
  ArrayXMap makeThreshold(index nBins, double x1, double y1, double x2,
                               double y2)
  {
    using namespace Eigen;
    ArrayXMap threshold(mMaskThreshBuf.data(),nBins);
    threshold = ArrayXd::Ones(nBins);
    index   kneeStart = static_cast<index>(std::floor(x1 * nBins));
    index   kneeEnd = static_cast<index>(std::floor(x2 * nBins));
    index   kneeLength = kneeEnd - kneeStart;
    threshold.segment(0, kneeStart) =
        ArrayXd::Constant(kneeStart, 10).pow(y1 / 20.0);
    threshold.segment(kneeStart, kneeLength) =
        ArrayXd::Constant(kneeLength, 10)
            .pow(ArrayXd::LinSpaced(kneeLength, y1, y2) / 20.0);
    threshold.segment(kneeEnd, nBins - kneeEnd) =
        ArrayXd::Constant(nBins - kneeEnd, 10).pow(y2 / 20.0);
    return threshold;
  }

  index                          mMaxBins;
  index                          mMaxHSize;
  Container<double>               mHBuf;
  Container<double>               mVBuf;
  Container<std::complex<double>> mFrameBuf;
  Container<double>               mPaddedBuf;
  Container<MedianFilter>         mHFilters;
  MedianFilter                   mVFilter;
  Container<double>               mHarmMaskBuf;
  Container<double>               mPercMaskBuf;
  Container<double>               mResMaskBuf;
  Container<double>               mMaskNormBuf;
  Container<double>               mMaskThreshBuf;
  bool                           mInitialized{false};
};
} // namespace algorithm
} // namespace fluid
