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
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class MelBands
{
public:
  /*static inline double mel2hz(double x) {
      return 700.0 * (exp(x / 1127.01048) - 1.0);
    }*/

  static inline double hz2mel(double x)
  {
    return 1127.01048 * log(x / 700.0 + 1.0);
  }

  void init(double lo, double hi, int nBands, int nBins, double sampleRate,
            bool logOutput)
  {

    using namespace Eigen;
    assert(hi > lo);
    assert(nBands > 1);
    mLo = lo;
    mHi = hi;
    mBands = nBands;
    mSampleRate = sampleRate;
    mBins = nBins;
    ArrayXd melFreqs = ArrayXd::LinSpaced(mBands + 2, hz2mel(lo), hz2mel(hi));
    melFreqs = 700.0 * ((melFreqs / 1127.01048).exp() - 1.0);
    mFilters = MatrixXd::Zero(mBands, mBins);
    ArrayXd fftFreqs = ArrayXd::LinSpaced(mBins, 0, mSampleRate / 2.0);
    ArrayXd melD =
        (melFreqs.segment(0, mBands + 1) - melFreqs.segment(1, mBands + 1))
            .abs();
    ArrayXXd ramps = melFreqs.replicate(1, mBins);
    ramps.rowwise() -= fftFreqs.transpose();
    for (int i = 0; i < mBands; i++)
    {
      ArrayXd lower = -ramps.row(i) / melD(i);
      ArrayXd upper = ramps.row(i + 2) / melD(i + 1);
      mFilters.row(i) = lower.min(upper).max(0);
    }
    // ArrayXd enorm =
    //     2.0 / (melFreqs.segment(2, mBands) - melFreqs.segment(0, mBands));
    // mFilters = (mFilters.array().colwise() *= enorm).matrix();
    // mOutputBuffer = ArrayXd::Zero(mBands);
    mLogOutput = logOutput;
  }

  void processFrame(const RealVectorView in, RealVectorView out)
  {
    using namespace Eigen;
    double const epsilon = std::numeric_limits<double>::epsilon();
    assert(in.size() == mBins);
    ArrayXd frame = _impl::asEigen<Eigen::Array>(in);
    ArrayXd result = (mFilters * frame.square().matrix()).array();
    if (mLogOutput) result = 10 * result.max(3.9810717055349695e-15).log10();
    out = _impl::asFluid(result);
  }

  double mLo{20.0};
  double mHi{20000.0};
  int    mBins{513};
  int    mBands{40};
  double mSampleRate{44100.0};
  bool   mLogOutput{false};

  Eigen::MatrixXd mFilters;

private:
  // ArrayXd mOutputBuffer;
};
}; // namespace algorithm
}; // namespace fluid
