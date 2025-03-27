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
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class MelBands
{
public:
  MelBands(
      index maxBands, index maxFFT, Allocator& alloc = FluidDefaultAllocator())
      : mFilters(maxBands, maxFFT / 2 + 1, alloc)
  {}

  /*static inline double mel2hz(double x) {
      return 700.0 * (exp(x / 1127.01048) - 1.0);
    }*/

  static inline double hz2mel(double x)
  {
    return 1127.01048 * std::log(x / 700.0 + 1.0);
  }

  void init(double lo, double hi, index nBands, index nBins, double sampleRate,
      index windowSize, Allocator& alloc = FluidDefaultAllocator())
  {

    using namespace Eigen;
    assert(hi > lo);
    assert(nBands > 1);
    assert(nBins <= mFilters.cols());
    mScale1 = 1.0 / (windowSize / 4.0); // scale to original amplitude
    index fftSize = 2 * (nBins - 1);

    mScale2 = 1.0 / (2.0 * double(fftSize) / windowSize);
    ScopedEigenMap<ArrayXd> melFreqs(nBands + 2, alloc);
    melFreqs = ArrayXd::LinSpaced(nBands + 2, hz2mel(lo), hz2mel(hi));
    melFreqs = 700.0 * ((melFreqs / 1127.01048).exp() - 1.0);
    //    mFilters = mFiltersStorage.block(0, 0, nBands, nBins);
    mFilters.topLeftCorner(nBands, nBins).setZero();
    ScopedEigenMap<ArrayXd> fftFreqs(nBins, alloc);
    fftFreqs = ArrayXd::LinSpaced(nBins, 0, sampleRate / 2.0);
    ScopedEigenMap<ArrayXd> melD(nBands + 1, alloc);
    melD = (melFreqs.segment(0, nBands + 1) - melFreqs.segment(1, nBands + 1))
               .abs();
    ScopedEigenMap<ArrayXXd> ramps(melFreqs.rows(), nBins, alloc);
    ramps = melFreqs.replicate(1, nBins);
    ramps.rowwise() -= fftFreqs.transpose();

    ScopedEigenMap<ArrayXd> lower(nBins, alloc);
    ScopedEigenMap<ArrayXd> upper(nBins, alloc);
    for (index i = 0; i < nBands; i++)
    {
      lower = -ramps.row(i) / melD(i);
      upper = ramps.row(i + 2) / melD(i + 1);
      mFilters.row(i).head(nBins) = lower.min(upper).max(0);
    }
    mNBands = nBands;
    mNBins = nBins;
  }

  void processFrame(const RealVectorView in, RealVectorView out, bool magNorm,
                    bool usePower, bool logOutput, Allocator&)
  {
    using namespace Eigen;

    FluidEigenMap<Array> frame = _impl::asEigen<Array>(in);
    FluidEigenMap<Array> result = _impl::asEigen<Array>(out);

    if (magNorm) frame = frame * mScale1;
    double energy = frame.sum() * mScale2;
    if (usePower) frame = frame.square();

    result.matrix().noalias() =
        (mFilters.topLeftCorner(mNBands, mNBins) * frame.matrix());

    if (magNorm) { result = result * energy / std::max(epsilon, result.sum()); }

    if (logOutput) result = 20 * result.max(epsilon).log10();
  }

  double mScale1{1.0};
  double mScale2{1.0};

private:
  ScopedEigenMap<Eigen::MatrixXd> mFilters;
  index                           mNBands;
  index                           mNBins;
};
} // namespace algorithm
} // namespace fluid
