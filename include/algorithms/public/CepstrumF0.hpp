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

#include "DCT.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PeakDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Eigen>
#include <cmath>

namespace fluid {
namespace algorithm {

class CepstrumF0
{

public:
  using ArrayXd = Eigen::ArrayXd;

  CepstrumF0(index maxSize) : mCepstrumStorage(maxSize) {}

  void init(index size)
  {
    // avoid allocation of maxSize^2 at constructor
    mDCT = DCT(size, size);
    mDCT.init(size, size);

    mCepstrum = mCepstrumStorage.segment(0, size);
    mCepstrum.setZero();
  }

  void processFrame(const RealVectorView& input, RealVectorView output,
                    double minFreq, double maxFreq, double sampleRate)
  {
    using namespace Eigen;
    using namespace std;
    PeakDetection pd;

    ArrayXd mag = _impl::asEigen<Array>(input);
    ArrayXd logMag = mag.max(epsilon).log();
    double  pitch = 0;
    double  confidence = 0;
    index   minBin = min(lrint(sampleRate / maxFreq),  mag.size());
    index   maxBin = min(lrint(sampleRate / minFreq), mag.size());

    mDCT.processFrame(logMag, mCepstrum);

    if (maxBin > minBin)
    {
      auto seg = mCepstrum.segment(minBin, maxBin - minBin);
      auto vec = pd.process(mCepstrum.segment(minBin, maxBin - minBin), 1, seg.minCoeff());
      if (vec.size() > 0)
      {
        pitch = sampleRate / (vec[0].first + minBin);
        confidence = vec[0].second / mCepstrum[0];
      }
    }
    output(0) = pitch;
    output(1) = min(abs(confidence), 1.0);
  }

private:
  DCT     mDCT{0, 0};
  ArrayXd mCepstrumStorage;
  ArrayXd mCepstrum;
};
} // namespace algorithm
} // namespace fluid
