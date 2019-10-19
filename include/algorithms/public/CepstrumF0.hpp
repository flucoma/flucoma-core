/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/DCT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PeakDetection.hpp"

#include <Eigen/Eigen>

namespace fluid {
namespace algorithm {

class CepstrumF0
{

public:
  using ArrayXd = Eigen::ArrayXd;

  void init(int size)
  {
    mDCT.init(size, size);
    mCepstrum = ArrayXd(size);
  }

  void processFrame(const RealVectorView& input, RealVectorView output,
                    double minFreq, double maxFreq, double sampleRate)
  {
    using namespace Eigen;
    PeakDetection pd;

    ArrayXd mag = _impl::asEigen<Array>(input);
    ArrayXd logMag = mag.max(epsilon).log();
    double  pitch = sampleRate / minFreq;
    double  confidence = 0;
    int     minBin = std::round(sampleRate / maxFreq);
    int     maxBin = std::round(sampleRate / minFreq);

    mDCT.processFrame(logMag, mCepstrum);
    
    if (maxBin > minBin && maxBin < mCepstrum.size())
    {
      auto vec = pd.process(mCepstrum.segment(minBin, maxBin - minBin), 1);
      if (vec.size() > 0)
      {
        pitch = sampleRate / (vec[0].first + minBin);
        confidence = vec[0].second / mCepstrum[0];
      }
    }
    output(0) = pitch;
    output(1) = std::min(std::abs(confidence), 1.0);
  }

private:
  DCT     mDCT;
  ArrayXd mCepstrum;
};
} // namespace algorithm
} // namespace fluid
