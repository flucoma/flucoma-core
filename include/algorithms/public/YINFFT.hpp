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

#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PeakDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class YINFFT
{

public:
  void processFrame(const RealVectorView& input, RealVectorView output,
                    double minFreq, double maxFreq, double sampleRate)
  {
    using namespace Eigen;
    PeakDetection pd;
    ArrayXd       mag = _impl::asEigen<Array>(input);
    ArrayXd       squareMag = mag.square();
    index         nBins = mag.size();
    FFT           fft(2 * (mag.size() - 1));
    double        squareMagSum = 2 * squareMag.sum();
    ArrayXd       squareMagSym(2 * (nBins - 1));
    squareMagSym << squareMag[0], squareMag.segment(1, nBins - 1),
        squareMag.segment(1, nBins - 2).reverse();
    ArrayXcd squareMagFFT = fft.process(squareMagSym);
    ArrayXd  yin = squareMagSum - squareMagFFT.real();
    if (maxFreq == 0) maxFreq = 1;
    if (minFreq == 0) minFreq = 1;
    yin(0) = 1;
    double tmpSum = 0;
    for (index i = 1; i < nBins; i++)
    {
      tmpSum += yin(i);
      yin(i) *= i / tmpSum;
    }
    double pitch = sampleRate / minFreq;
    double pitchConfidence = 0;
    if (tmpSum > 0)
    {
      ArrayXd yinFlip = -yin;
      // segment from max to min freq
      index minBin = std::lrint(sampleRate / maxFreq);
      index maxBin = std::lrint(sampleRate / minFreq);
      if (minBin > yinFlip.size() - 1) minBin = yinFlip.size() - 1;
      if (maxBin > yinFlip.size() - minBin - 1)
        maxBin = yinFlip.size() - minBin - 1;
      if (maxBin > minBin)
      {
        yinFlip = yinFlip.segment(minBin, maxBin - minBin);
        auto vec = pd.process(yinFlip, 1, yinFlip.minCoeff());
        if (vec.size() > 0)
        {
          pitch = sampleRate / (minBin + vec[0].first);
          pitchConfidence = 1 + vec[0].second;
        }
      }
    }
    output(0) = pitch;
    output(1) = pitchConfidence;
  }
};
} // namespace algorithm
} // namespace fluid
