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
#include "../util/AlgorithmUtils.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PartialTracking.hpp"
#include "../util/PeakDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>
#include <queue>

namespace fluid {
namespace algorithm {


class SineFeature
{
  using ArrayXd = Eigen::ArrayXd;
  using VectorXd = Eigen::VectorXd;
  using ArrayXcd = Eigen::ArrayXcd;
  template <typename T>
  using vector = rt::vector<T>;

public:
  SineFeature(Allocator&) {}

  void init(index windowSize, index fftSize)
  {
    mBins = fftSize / 2 + 1;
    mScale = 1.0 / (windowSize / 4.0); // scale to original amplitude
    mInitialized = true;
  }

  index processFrame(const ComplexVectorView in, RealVectorView freqOut,
                     RealVectorView magOut, double sampleRate,
                     double detectionThreshold, index sortBy, Allocator& alloc)
  {
    assert(mInitialized);
    using namespace Eigen;
    index                    fftSize = 2 * (mBins - 1);
    ScopedEigenMap<ArrayXcd> frame(in.size(), alloc);
    frame = _impl::asEigen<Array>(in);

    ScopedEigenMap<ArrayXd> mag(in.size(), alloc);
    mag = frame.abs().real();
    mag = mag * mScale;
    ScopedEigenMap<ArrayXd> logMagIn(in.size(), alloc);
    logMagIn = 20 * mag.max(epsilon).log10();

    auto tmpPeaks =
        mPeakDetection.process(logMagIn, 0, detectionThreshold, true, sortBy);

    index maxNumOut = std::min<index>(freqOut.size(), asSigned(tmpPeaks.size()));

    double ratio = sampleRate / fftSize;
    std::transform(tmpPeaks.begin(), tmpPeaks.begin() + maxNumOut,
                   freqOut.begin(),
                   [ratio](auto peak) { return peak.first * ratio; });

    std::transform(tmpPeaks.begin(), tmpPeaks.begin() + maxNumOut,
                   magOut.begin(), [](auto peak) { return peak.second; });

    return maxNumOut;
  }

  bool initialized() const { return mInitialized; }

private:
  PeakDetection mPeakDetection;
  index         mBins{513};
  double        mScale{1.0};
  bool          mInitialized{false};
};
} // namespace algorithm
} // namespace fluid
