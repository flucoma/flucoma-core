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

#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PeakDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class YINFFT
{

public:
    
  YINFFT(index maxInputSize, Allocator& alloc = FluidDefaultAllocator())
  : mFFT(2 * maxInputSize - 1, alloc)
  {}
  
  void processFrame(const RealVectorView& input, RealVectorView output,
      double minFreq, double maxFreq, double sampleRate,
      Allocator& alloc = FluidDefaultAllocator())
  {
    using namespace Eigen;
    PeakDetection pd;
    //    ScopedEigenMap<ArrayXd>  mag = _;
    ScopedEigenMap<ArrayXd> squareMag(input.size(), alloc);
    squareMag = _impl::asEigen<Array>(input).square();

    index  nBins = input.size();
    double squareMagSum = 2 * squareMag.sum();

    mFFT.resize(2 * (nBins - 1));
      
    ScopedEigenMap<ArrayXd> squareMagSym(2 * (nBins - 1), alloc);
    squareMagSym << squareMag[0], squareMag.segment(1, nBins - 1),
        squareMag.segment(1, nBins - 2).reverse();

    Eigen::Map<ArrayXcd>    squareMagFFT = mFFT.process(squareMagSym);
    ScopedEigenMap<ArrayXd> yin(squareMagFFT.size(), alloc);
    yin = squareMagSum - squareMagFFT.real();

    if (maxFreq == 0) maxFreq = 1;
    if (minFreq == 0) minFreq = 1;
    yin(0) = 1;
    double tmpSum = 0;
    for (index i = 1; i < nBins; i++)
    {
      tmpSum += yin(i);
      yin(i) *= i / tmpSum;
    }
    double pitch = 0;
    double pitchConfidence = 0;
    if (tmpSum > 0)
    {
      ScopedEigenMap<ArrayXd> yinFlip(yin.size(), alloc);
      yinFlip = -yin;
      // segment from max to min freq
      index minBin = std::lrint(sampleRate / maxFreq);
      index maxBin = std::lrint(sampleRate / minFreq);
      if (minBin > yinFlip.size() - 1) minBin = yinFlip.size() - 1;
      if (maxBin > yinFlip.size() - minBin - 1)
        maxBin = yinFlip.size() - minBin - 1;
      if (maxBin > minBin)
      {
        //        yinFlip = yinFlip.segment(minBin, maxBin - minBin).eval();
        auto yinSeg = yinFlip.segment(minBin, maxBin - minBin);
        auto vec = pd.process(yinSeg, 1, yinSeg.minCoeff(), true, true, alloc);
        if (vec.size() > 0)
        {
          pitch = sampleRate / (minBin + vec[0].first);
          pitchConfidence = std::max(1. + vec[0].second, 0.);
        }
      }
    }
    output(0) = pitch;
    output(1) = pitchConfidence;
  }
    
  FFT mFFT;
};
} // namespace algorithm
} // namespace fluid
