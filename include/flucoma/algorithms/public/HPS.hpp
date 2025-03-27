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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class HPS
{

public:
  void processFrame(const RealVectorView& input, RealVectorView output,
                    index nHarmonics, double minFreq, double maxFreq,
                    double sampleRate, Allocator& alloc)
  {
    using namespace Eigen;
    using namespace std;

    ArrayXd::Index maxIndex;

    ScopedEigenMap<ArrayXd> mag(input.size(), alloc);
    mag = _impl::asEigen<Array>(input);
    
    ScopedEigenMap<ArrayXd> hps(mag.size(), alloc);
    hps = mag;
    index   nBins = mag.size();
    double  binHz = sampleRate / ((nBins - 1) * 2);
    index   minBin = lrint(minFreq / binHz);
    index   maxBin = lrint(maxFreq / binHz);
    double  f0 = 0;
    double  confidence = 0;
    double hpsSum = 0;

    ScopedEigenMap<ArrayXd> h(nBins, alloc);
    ScopedEigenMap<ArrayXd> hp(nBins, alloc);
    for (index i = 2; i < nHarmonics; i++)
    {
      index   hBins = nBins / i;
      for (index j = 0; j < hBins; j++) h.head(hBins)(j) = mag(j * i);
      hp.setZero();
      hp.head(hBins) = h.head(hBins);
      hps = hps * hp;
    }
    hpsSum = hps.sum();

    if (maxBin > minBin &&  hpsSum > 0)
    {
      double maxVal = hps.segment(minBin, maxBin - minBin).maxCoeff(&maxIndex);
      confidence = maxVal / hpsSum;
      f0 = (minBin + maxIndex) * binHz;
    }
    output(0) = f0;
    output(1) = confidence;
  }
};
} // namespace algorithm
} // namespace fluid
