/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

// Based on Ellis, Daniel P.W. "Chroma feature analysis and synthesis"
// https://www.ee.columbia.edu/~dpwe/resources/matlab/chroma-ansyn/

#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class ChromaFilterBank
{
public:
  ChromaFilterBank(index maxBins, index maxFFT)
      : mFiltersStorage(maxBins, maxFFT / 2 + 1)
  {}


  void init(index nChroma, index nBins, double ref, double sampleRate)
  {
    using namespace Eigen;
    const double log2E = 1.44269504088896340736;
    index fftSize = 2 * (nBins - 1);
    ArrayXd freqs = ArrayXd::LinSpaced(fftSize, 0, sampleRate);
    freqs = nChroma * (freqs / (ref / 16)).log() * log2E;
    freqs[0] = freqs[1] - 1.5 * nChroma;

    ArrayXd widths =  ArrayXd::Ones(fftSize);
    widths.segment(0, fftSize - 1) =
        freqs.segment(1, fftSize - 1) -
        freqs.segment(0, fftSize - 1);
    widths = widths.max(1.0);
    ArrayXXd diffs =
      freqs.replicate(1, nChroma).transpose() -
      ArrayXd::LinSpaced(nChroma, 0, nChroma - 1).transpose().replicate(fftSize, 1).transpose();
    index halfChroma = std::lrint(nChroma / 2);
    ArrayXXd remainder =  diffs.unaryExpr([&](const double x){
        return std::fmod(x + 10* nChroma + halfChroma, nChroma) - halfChroma;
    });
    MatrixXd filters = (-0.5 * (2 * remainder / widths.replicate(1, nChroma).transpose()).square()).exp();
    filters = filters.block(0, 0, nChroma, nBins);
    filters.colwise().normalize();
    mFiltersStorage.setZero();
    mFiltersStorage.block(0, 0, nChroma, nBins) = filters;
    mNChroma = nChroma;
    mNBins = nBins;
    mScale = 2.0 / (fftSize * mNChroma);
    mSampleRate = sampleRate;
  }

  void processFrame(const RealVectorView in, RealVectorView out,
    double minFreq = 0, double maxFreq = -1, index normalize = 0)
  {
    using namespace Eigen;
    using namespace std;
    ArrayXd frame = _impl::asEigen<Eigen::Array>(in);
    Eigen::Ref<Eigen::MatrixXd> filters = mFiltersStorage.block(0, 0, mNChroma, mNBins);

    if(minFreq != 0 || maxFreq != -1){
        maxFreq = (maxFreq == -1) ? (mSampleRate / 2) : min(maxFreq, mSampleRate / 2);
        double  binHz = mSampleRate / ((mNBins - 1) * 2.);
        index   minBin = minFreq == 0? 0 : ceil(minFreq / binHz);
        index   maxBin =
            min(static_cast<index>(floorl(maxFreq / binHz)), (mNBins - 1));
        frame.segment(0, minBin).setZero();
        frame.segment(maxBin, frame.size() - maxBin).setZero();
    }

    ArrayXd result = mScale * (filters * frame.square().matrix()).array();

    if (normalize > 0) {
      double norm = normalize == 1? result.sum() : result.maxCoeff();
      result = result / std::max(norm, epsilon);
    }
    out <<= _impl::asFluid(result);
  }

  index mNChroma;
  index mNBins;
  double mScale;
  double mSampleRate;
  Eigen::MatrixXd mFiltersStorage;
};
} // namespace algorithm
} // namespace fluid
