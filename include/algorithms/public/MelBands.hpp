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

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class MelBands
{
public:
  MelBands(index maxBands, index maxFFT)
      : mFiltersStorage(maxBands, maxFFT / 2 + 1)
  {}

  /*static inline double mel2hz(double x) {
      return 700.0 * (exp(x / 1127.01048) - 1.0);
    }*/

  static inline double hz2mel(double x)
  {
    return 1127.01048 * std::log(x / 700.0 + 1.0);
  }

  void init(double lo, double hi, index nBands, index nBins, double sampleRate,
            index windowSize)
  {

    using namespace Eigen;
    assert(hi > lo);
    assert(nBands > 1);
    mScale1 = 1.0 / (windowSize / 4.0); // scale to original amplitude
    index fftSize = 2 * (nBins - 1);
    mScale2 = 1.0 / (2.0 * double(fftSize) / windowSize);
    ArrayXd melFreqs = ArrayXd::LinSpaced(nBands + 2, hz2mel(lo), hz2mel(hi));
    melFreqs = 700.0 * ((melFreqs / 1127.01048).exp() - 1.0);
    mFilters = mFiltersStorage.block(0, 0, nBands, nBins);
    mFilters.setZero();
    ArrayXd fftFreqs = ArrayXd::LinSpaced(nBins, 0, sampleRate / 2.0);
    ArrayXd melD =
        (melFreqs.segment(0, nBands + 1) - melFreqs.segment(1, nBands + 1))
            .abs();
    ArrayXXd ramps = melFreqs.replicate(1, nBins);
    ramps.rowwise() -= fftFreqs.transpose();
    for (index i = 0; i < nBands; i++)
    {
      ArrayXd lower = -ramps.row(i) / melD(i);
      ArrayXd upper = ramps.row(i + 2) / melD(i + 1);
      mFilters.row(i) = lower.min(upper).max(0);
    }
  }

  void processFrame(const RealVectorView in, RealVectorView out, bool magNorm, bool usePower, bool logOutput)
  {
    using namespace Eigen;

    ArrayXd frame = _impl::asEigen<Eigen::Array>(in);
    if (magNorm) frame = frame * mScale1;
    ArrayXd result;
    if (usePower) { result = (mFilters * frame.square().matrix()).array(); }
    else
    {
      result = (mFilters * frame.matrix()).array();
    }
    if (magNorm)
    {
      double energy = frame.sum() * mScale2;
      result = result * energy / std::max(epsilon, result.sum());
    }

    if (logOutput) result = 10 * result.max(epsilon).log10();
    out = _impl::asFluid(result);
  }

  double mScale1{1.0};
  double mScale2{1.0};

  Eigen::MatrixXd mFilters;
  Eigen::MatrixXd mFiltersStorage;
};
} // namespace algorithm
} // namespace fluid
