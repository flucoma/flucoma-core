/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

// Zdenek Prusa and Peter L. Soendergaard,
// Real-time spectrogram inversion using phase gradient heap integration
// Proceedings of DAFX 2016

#pragma once

#include "AlgorithmUtils.hpp"
#include "FluidEigenMappings.hpp"
#include "../public/STFT.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class RTPGHI
{

public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXcd = Eigen::ArrayXcd;

  RTPGHI(index maxFFTSize, Allocator& alloc)
      : mMaxBins(maxFFTSize / 2 + 1),
        mBinIndices(mMaxBins, alloc),
        mPrevMag(mMaxBins, alloc),
        mPrevLogMag(mMaxBins, alloc),
        mPrevPrevLogMag(mMaxBins, alloc),
        mPrevPhase(mMaxBins, alloc),
        mPrevPrevPhase(mMaxBins, alloc),
        mPrevPhaseDeltaT(mMaxBins, alloc)
  {}


  void init(index fftSize)
  {
    mBins = fftSize / 2 + 1;
    assert(mBins <= mMaxBins);
    mPrevMag.head(mBins).setZero();
    mPrevLogMag.head(mBins).setZero();
    mPrevPrevLogMag.head(mBins).setZero();
    mPrevPhase.head(mBins).setZero();
    mPrevPrevPhase.head(mBins).setZero();
    mPrevPhaseDeltaT.head(mBins).setZero();
    mBinIndices.head(mBins) = ArrayXd::LinSpaced(mBins, 0, mBins - 1);
  }

  void processFrame(RealVectorView in, ComplexVectorView out, index winSize,
      index fftSize, index hopSize, double tolerance, Allocator& alloc)
  {
    using namespace Eigen;
    using namespace _impl;
    using namespace std;
    using namespace std::complex_literals;
    double gamma = 0.25645 * pow(winSize, 2); // assumes Hann window
    assert(in.size() == mBins);
    ScopedEigenMap<ArrayXd> mag(mBins, alloc);
    mag = asEigen<Array>(in);
    ScopedEigenMap<ArrayXd> logMag(mBins, alloc);
    logMag = mag.max(epsilon).log();
    ScopedEigenMap<ArrayXd> futureLogMag(mBins, alloc);
    futureLogMag = logMag;
    ScopedEigenMap<ArrayXd> currentLogMag(mBins, alloc);
    currentLogMag = mPrevLogMag.head(mBins);
    ScopedEigenMap<ArrayXd> prevLogMag(mBins, alloc);
    prevLogMag = mPrevPrevLogMag.head(mBins);

    ScopedEigenMap<ArrayXd> phaseDeltaT =
        getPhaseDeltaT(currentLogMag, gamma, fftSize, hopSize, alloc);
    ScopedEigenMap<ArrayXd> phaseDeltaF = getPhaseDeltaF(
        prevLogMag, futureLogMag, gamma, fftSize, hopSize, alloc);
    double absTol =
        log(tolerance) + max(currentLogMag.maxCoeff(), prevLogMag.maxCoeff());
    ScopedEigenMap<ArrayXd> todo(currentLogMag.size(), alloc);
    todo = (currentLogMag > absTol).cast<double>();
    index                   numTodo = static_cast<index>(todo.sum());
    ScopedEigenMap<ArrayXd> phaseEst(mBins, alloc);
    phaseEst = pi + ArrayXd::Random(mBins) * pi;

    rt::vector<pair<double, index>> heap(alloc);
    heap.reserve(asUnsigned(mBins));
    for (index i = 0; i < mBins; i++)
    {
      if (prevLogMag(i) > absTol) heap.push_back({prevLogMag(i), i});
    }
    make_heap(heap.begin(), heap.end());

    while (numTodo > 0 && heap.size() > 0)
    {
      pop_heap(heap.begin(), heap.end());
      index _m = heap.back().second;
      heap.pop_back();

      // use indices 0..mBins for prev frame
      // mBins ... 2 * mBins  for current frame
      if (_m < mBins && todo[_m] > 0)
      {
        index m = _m;
        phaseEst[m] =
            mPrevPhase[m] + 0.5 * (phaseDeltaT[m] + mPrevPhaseDeltaT[m]);
        heap.push_back({currentLogMag[m], m + mBins});
        push_heap(heap.begin(), heap.end());
        todo[m] = 0;
        numTodo--;
      }
      else if (_m >= mBins)
      {
        index m = _m - mBins;
        if (m < mBins - 1 && todo[m + 1] > 0)
        {
          phaseEst[m + 1] =
              phaseEst[m] + 0.5 * (phaseDeltaF[m] + phaseDeltaF[m + 1]);
          heap.push_back({currentLogMag[m + 1], _m + 1});
          push_heap(heap.begin(), heap.end());
          todo[m + 1] = 0;
          numTodo--;
        }
        if (m > 0 && todo[m - 1] > 0)
        {
          phaseEst[m - 1] =
              phaseEst[m] - 0.5 * (phaseDeltaF[m] + phaseDeltaF[m - 1]);
          heap.push_back({currentLogMag[m - 1], _m - 1});
          push_heap(heap.begin(), heap.end());
          todo[m - 1] = 0;
          numTodo--;
        }
      }
    }

    ScopedEigenMap<ArrayXd> finalPhase(mBins, alloc);
    finalPhase =
        phaseEst - ArrayXd::LinSpaced(mBins, 0, 1) * pi * (winSize - 1) / 2;
    ScopedEigenMap<ArrayXcd> result(mBins, alloc);
    result = mPrevMag.head(mBins) * (1i * finalPhase).exp();
    mPrevPrevLogMag.head(mBins) = mPrevLogMag.head(mBins);
    mPrevLogMag.head(mBins) = logMag;
    mPrevPhase.head(mBins) = phaseEst;
    mPrevPhaseDeltaT.head(mBins) = phaseDeltaT;
    mPrevMag.head(mBins) = mag;
    asEigen<Eigen::Array>(out) = result;
  }

private:
  ScopedEigenMap<ArrayXd> getPhaseDeltaT(Eigen::Ref<ArrayXd> logMag,
      double gamma, index fftSize, index hopSize, Allocator& alloc)
  {
    ScopedEigenMap<ArrayXd> deltaT(mBins, alloc);
    deltaT.setZero();
    deltaT.segment(1, mBins - 2) =
        logMag.segment(2, mBins - 2) - logMag.segment(0, mBins - 2);
    deltaT = deltaT * 0.5 * hopSize * fftSize / gamma;
    deltaT = deltaT + twoPi * hopSize * mBinIndices / fftSize;
    return deltaT;
  }

  ScopedEigenMap<ArrayXd> getPhaseDeltaF(Eigen::Ref<ArrayXd> prevLogMag,
      Eigen::Ref<ArrayXd> nextLogMag, double gamma, index fftSize,
      index hopSize, Allocator& alloc)
  {
    ScopedEigenMap<ArrayXd> result(prevLogMag.size(), alloc);
    result = 0.5 * (nextLogMag - prevLogMag) * (-gamma / (hopSize * fftSize));
    return result;
  }

  index                   mMaxBins;
  index                   mBins;
  ScopedEigenMap<ArrayXd> mBinIndices;
  ScopedEigenMap<ArrayXd> mPrevMag;
  ScopedEigenMap<ArrayXd> mPrevLogMag;
  ScopedEigenMap<ArrayXd> mPrevPrevLogMag;
  ScopedEigenMap<ArrayXd> mPrevPhase;
  ScopedEigenMap<ArrayXd> mPrevPrevPhase;
  ScopedEigenMap<ArrayXd> mPrevPhaseDeltaT;
};
} // namespace algorithm
} // namespace fluid
