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
#include "../../data/TensorTypes.hpp"
#include <Eigen/Eigen>
#include <cmath>

namespace fluid {
namespace algorithm {

class SpectralShape
{

  using ArrayXd = Eigen::ArrayXd;

public:
  SpectralShape(Allocator& alloc) : mOutputBuffer(7, alloc) {}

  void processFrame(Eigen::Ref<ArrayXd> in, double sampleRate, double minFreq,
                    double maxFreq, double rolloffTarget, bool logFreq,
                    bool usePower, Allocator& alloc)
  {
    using namespace std;
    maxFreq = (maxFreq == -1) ? (sampleRate / 2) : min(maxFreq, sampleRate / 2);
    ScopedEigenMap<ArrayXd> mag(in.size(), alloc);
    mag = in.max(epsilon);
    index  nBins = mag.size();
    double binHz = sampleRate / ((nBins - 1) * 2.);
    index  minBin = static_cast<index>(ceil(minFreq / binHz));
    index  maxBin =
        min(static_cast<index>(floorl(maxFreq / binHz)), (nBins - 1));
    if (maxBin <= minBin)
    {
      mOutputBuffer.setZero();
      return;
    }

    if (logFreq && minBin == 0)
    {
      minBin = 1;
      mag(1) += mag(0);
    }

    index size = maxBin - minBin;

    ScopedEigenMap<ArrayXd> amp(size, alloc);
    amp = mag.segment(minBin, size);
    if (usePower) amp = amp.square();

    double                  ampSum = amp.sum();
    ScopedEigenMap<ArrayXd> freqs(size, alloc);
    freqs = ArrayXd::LinSpaced(size, minBin * binHz, maxBin * binHz);
    if (logFreq)
    {
      freqs = 69 + (12 * (freqs / 440).log() * log2E);
    } // MIDI cents

    double centroid = (amp * freqs).sum() / ampSum;
    double spread = (amp * (freqs - centroid).square()).sum() / ampSum;
    double skewness = (amp * (freqs - centroid).pow(3)).sum() /
                      (spread * sqrt(spread) * ampSum);
    double kurtosis =
        (amp * (freqs - centroid).pow(4)).sum() / (spread * spread * ampSum);

    double flatness = exp(amp.log().mean()) / amp.mean();
    double rolloff = maxBin - 1;
    double cumSum = 0;
    double target = ampSum * rolloffTarget / 100.0;
    for (index i = 0; cumSum <= target && i < amp.size(); i++)
    {
      cumSum += amp(i);
      if (cumSum >= target)
      {
        rolloff = (i == 0) ? freqs(i)
                           : freqs(i) - (freqs(i) - freqs(i - 1)) *
                                            (cumSum - target) / amp(i);
        break;
      }
    }
    double crest = amp.maxCoeff() / amp.mean();

    mOutputBuffer(0) = centroid;
    mOutputBuffer(1) = sqrt(spread);
    mOutputBuffer(2) = skewness;
    mOutputBuffer(3) = kurtosis;
    mOutputBuffer(4) = rolloff;
    mOutputBuffer(5) = 20 * log10(max(flatness, epsilon));
    mOutputBuffer(6) = 20 * log10(max(crest, epsilon));
  }

  void processFrame(const RealVectorView input, RealVectorView output,
                    double sampleRate, double minFreq, double maxFreq,
                    double rolloffTarget, bool logFreq, bool usePower,
                    Allocator& alloc)
  {
    assert(output.size() == 7);
    ScopedEigenMap<ArrayXd> in(input.size(), alloc);
    in = _impl::asEigen<Eigen::Array>(input);
    processFrame(in, sampleRate, minFreq, maxFreq, rolloffTarget, logFreq,
                 usePower, alloc);
    _impl::asEigen<Eigen::Array>(output) = mOutputBuffer;
  }

private:
  ScopedEigenMap<ArrayXd> mOutputBuffer;
};

} // namespace algorithm
} // namespace fluid
