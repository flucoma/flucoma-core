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

#include "WindowFuncs.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PartialTracking.hpp"
#include "../util/PeakDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>
#include <queue>

namespace fluid {
namespace algorithm {


class SineFeature 
{
  using ArrayXd = Eigen::ArrayXd;
  using VectorXd = Eigen::VectorXd;
  using ArrayXcd = Eigen::ArrayXcd;
  template <typename T>
  using vector = rt::vector<T>;

public:
  SineFeature(Allocator& alloc)
  {}

  void init(index windowSize, index fftSize)
  {
    mBins = fftSize / 2 + 1;
    mScale = 1.0 / (windowSize / 4.0); // scale to original amplitude
    mInitialized = true;
  }

  void processFrame(const ComplexVectorView in, RealVectorView freqOut,
                    RealVectorView magOut, index logFreq, index logMag,
                    double sampleRate, double detectionThreshold,
                    index sortBy, Allocator& alloc)
  {
    assert(mInitialized);
    using namespace Eigen;
    index                    fftSize = 2 * (mBins - 1);
    ScopedEigenMap<ArrayXcd> frame(in.size(), alloc);
    frame = _impl::asEigen<Array>(in);

    ScopedEigenMap<ArrayXd> mag(in.size(), alloc);
    mag = frame.abs().real();
    mag = mag * mScale;
    ScopedEigenMap<ArrayXd> logMagIn(in.size(), alloc);
    logMagIn = 20 * mag.max(epsilon).log10();

    auto tmpPeaks = mPeakDetection.process(logMagIn, 0, detectionThreshold, true, sortBy);
    
    index maxNumOut = std::min<index>(freqOut.size(),tmpPeaks.size());
    
    double ratio = sampleRate / fftSize;
    if (logFreq){
      ratio = ratio / 440.0;
      std::transform(tmpPeaks.begin(), tmpPeaks.begin() + maxNumOut, freqOut.begin(), [ratio](auto peak){return 69 + (12 * std::log2(peak.first * ratio));});
      freqOut(Slice(maxNumOut, freqOut.size() - maxNumOut)).fill(-999);//pad the size with "no-pitch" in MIDI;            
    } else {
      std::transform(tmpPeaks.begin(), tmpPeaks.begin() + maxNumOut, freqOut.begin(), [ratio](auto peak){return peak.first * ratio;});
      freqOut(Slice(maxNumOut, freqOut.size() - maxNumOut)).fill(0);//pad the size with "no-pitch" (0Hz);      
    }
    
    if (logMag) {
      std::transform(tmpPeaks.begin(),tmpPeaks.begin()+maxNumOut,magOut.begin(),[](auto peak){return peak.second;});
      magOut(Slice(maxNumOut, freqOut.size() - maxNumOut)).fill(-144);//pad the size with 'silence' in dB;
    } else {
      std::transform(tmpPeaks.begin(),tmpPeaks.begin()+maxNumOut,magOut.begin(),[](auto peak){return std::pow(10, (peak.second / 20));});
      magOut(Slice(maxNumOut, freqOut.size() - maxNumOut)).fill(0);//pad the size with silence;    
    }
  }

  bool initialized() { return mInitialized; }

private:
  PeakDetection           mPeakDetection;
  index                   mBins{513};
  double                  mScale{1.0};
  bool                    mInitialized{false};
};
} // namespace algorithm
} // namespace fluid
