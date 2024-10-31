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

#include "DCT.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PeakDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Eigen>
#include <cmath>

namespace fluid {
namespace algorithm {

class CepstrumF0
{

public:
  using ArrayXd = Eigen::ArrayXd;

  CepstrumF0(index maxSize, Allocator& alloc)
    : mDCT{maxSize, maxSize, alloc}, mCepstrum(maxSize, alloc) {}

  void init(index size, Allocator& alloc)
  {
    // avoid allocation of maxSize^2 at constructor
//    mDCT = DCT(size, size);
    mDCT.init(size, size, alloc);
//    mCepstrum = mCepstrumStorage.segment(0, size);
    mCepstrum.setZero();
  }

  void processFrame(const RealVectorView& input, RealVectorView output,
                    double minFreq, double maxFreq, double sampleRate, Allocator& alloc)
  {
    using namespace Eigen;
    using namespace std;
    PeakDetection pd;

//    ArrayXd mag = _impl::asEigen<Array>(input);
    ScopedEigenMap<ArrayXd> logMag(input.size(), alloc);
    logMag = _impl::asEigen<Array>(input).max(epsilon).log();
    
    double  pitch = 0;
    double  confidence = 0;
    index   minBin = min<index>(lrint(sampleRate / maxFreq), logMag.size());
    index   maxBin = min<index>(lrint(sampleRate / minFreq), logMag.size());

    mDCT.processFrame(logMag, mCepstrum);

    if (maxBin > minBin)
    {
      auto seg = mCepstrum.segment(minBin, maxBin - minBin);
      auto vec = pd.process(mCepstrum.segment(minBin, maxBin - minBin), 1,
                            seg.minCoeff(), true, true, alloc);
      if (vec.size() > 0)
      {
        pitch = sampleRate / (vec[0].first + minBin);
        confidence = vec[0].second / mCepstrum[0];
      }
    }
    output(0) = pitch;
    output(1) = min(abs(confidence), 1.0);
  }

private:
  DCT     mDCT;
  ScopedEigenMap<ArrayXd> mCepstrum;
};
} // namespace algorithm
} // namespace fluid
