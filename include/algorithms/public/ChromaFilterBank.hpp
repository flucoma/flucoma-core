/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
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
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class ChromaFilterBank
{
public:
  ChromaFilterBank(index maxBins, index maxFFT, Allocator& alloc)
        : mMaxChroma(maxBins), mMaxFFT(maxFFT), mFilters(maxBins, maxFFT/2 + 1, alloc)
  {}


  void init(index nChroma, index nBins, double ref, double sampleRate, Allocator& alloc)
  {
    using namespace Eigen;
    constexpr double log2E = 1.44269504088896340736;
    index fftSize = 2 * (nBins - 1);
    assert(fftSize <= mMaxFFT);
    assert(nChroma <= mMaxChroma);
    
    ScopedEigenMap<Eigen::ArrayXd> freqs(fftSize, alloc);
    freqs = ArrayXd::LinSpaced(fftSize, 0, sampleRate);
    freqs = nChroma * (freqs / (ref / 16)).log() * log2E;
    freqs[0] = freqs[1] - 1.5 * nChroma;

    ScopedEigenMap<Eigen::ArrayXd> widths(fftSize, alloc);
    widths =  ArrayXd::Ones(fftSize);
    widths.segment(0, fftSize - 1) =
        freqs.segment(1, fftSize - 1) -
        freqs.segment(0, fftSize - 1);
    widths = widths.max(1.0);
    
    ScopedEigenMap<Eigen::ArrayXXd> diffs(nChroma, fftSize, alloc);
    diffs =
      freqs.replicate(1, nChroma).transpose() -
      ArrayXd::LinSpaced(nChroma, 0, nChroma - 1).transpose().replicate(fftSize, 1).transpose();
    index halfChroma = std::lrint(nChroma / 2);
    
    ScopedEigenMap<ArrayXXd> remainder(nChroma, fftSize, alloc);
    remainder =  diffs.unaryExpr([&](const double x){
        return std::fmod(x + 10* nChroma + halfChroma, nChroma) - halfChroma;
    });
    
    mFilters.topLeftCorner(nChroma,nBins) = (-0.5 * (2 * remainder / widths.replicate(1, nChroma)
    .transpose())
    .square())
    .exp()
    .block(0, 0, nChroma, nBins);

    ScopedEigenMap<ArrayXd> colNorms(nBins, alloc);
    colNorms = mFilters.topLeftCorner(nChroma, nBins).colwise().norm();
    mFilters.topLeftCorner(nChroma, nBins).array().rowwise() /=
        colNorms.transpose();

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
    
    FluidEigenMap<Eigen::Array> frame = _impl::asEigen<Eigen::Array>(in);
    
    if(minFreq != 0 || maxFreq != -1){
        maxFreq = (maxFreq == -1) ? (mSampleRate / 2) : min(maxFreq, mSampleRate / 2);
        double  binHz = mSampleRate / ((mNBins - 1) * 2.);
        index   minBin =
            minFreq == 0 ? 0 : static_cast<index>(ceil(minFreq / binHz));
        index   maxBin =
            min(static_cast<index>(floorl(maxFreq / binHz)), (mNBins - 1));
          
          using Eigen::seq;
          using Eigen::seqN;
          frame(seq(0,minBin),seqN(0,1)).setZero();
          frame(seqN(maxBin, frame.size() - maxBin), seqN(0,1)).setZero(); 
    }
    frame = frame.square();

    FluidEigenMap<Eigen::Array> result = _impl::asEigen<Eigen::Array>(out);
    result.matrix().noalias() =
        mFilters.topLeftCorner(mNChroma, mNBins) * frame.matrix();
    result *= mScale;

    if (normalize > 0) {
      double norm = normalize == 1? result.sum() : result.maxCoeff();
      result = result / std::max(norm, epsilon);
    }
  }

  index mNChroma;
  index mNBins;
  double mScale;
  double mSampleRate;
  index mMaxChroma;
  index mMaxFFT;
  ScopedEigenMap<Eigen::MatrixXd> mFilters;
};
} // namespace algorithm
} // namespace fluid
