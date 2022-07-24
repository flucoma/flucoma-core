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

#include "FFT.hpp"
#include "FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Eigen>
#include <cmath>

namespace fluid {
namespace algorithm {

class TruePeak
{

  using ArrayXcd = Eigen::ArrayXcd;

public:
  TruePeak(index maxSize, Allocator& alloc)
    : mFFT(maxSize, alloc), mIFFT(maxSize * 4, alloc),
      mBuffer(0,alloc)
    {}

  void init(index size, double sampleRate, Allocator& alloc)
  {
    using namespace std;
    mSampleRate = sampleRate;
    mFFTSize = static_cast<index>(pow(2, ceil(log(size) / log(2))));
    mFactor = sampleRate < 96000 ? 4 : 2;
    mFFT.resize(mFFTSize);
    mIFFT.resize(mFFTSize * mFactor);
    mBuffer = ScopedEigenMap<ArrayXcd>((mFFTSize * mFactor / 2) + 1,alloc);
  }

  double processFrame(const RealVectorView& input, Allocator& alloc)
  {
    using namespace Eigen;
    if (mSampleRate >= 192000)
    {
      return _impl::asEigen<Array>(input).abs().maxCoeff();
    }
    else
    {
      ScopedEigenMap<ArrayXd> in(input.size(), alloc);
      in = _impl::asEigen<Array>(input);
      Eigen::Map<ArrayXcd> transform = mFFT.process(in);
      mBuffer.setZero();
      mBuffer.segment(0, transform.size()) = transform;
      Eigen::Map<ArrayXd> result = mIFFT.process(mBuffer);
//      ArrayXd scaled = result / mFFTSize;
      double peak = (result / mFFTSize).abs().maxCoeff();
      return peak;
    }
  }

private:
  FFT      mFFT;
  IFFT     mIFFT;
  
  double   mSampleRate{44100.0};
  index    mFactor{4};
  index    mFFTSize{1024};
  ScopedEigenMap<ArrayXcd> mBuffer;
};
} // namespace algorithm
} // namespace fluid
