/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
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

  void init(index fftSize)
  {
    mBins = fftSize / 2 + 1;
    mPrevMag = ArrayXd::Zero(mBins);
    mPrevLogMag = ArrayXd::Zero(mBins);
    mPrevPrevLogMag = ArrayXd::Zero(mBins);
    mPrevPhase = ArrayXd::Zero(mBins);
    mPrevPrevPhase = ArrayXd::Zero(mBins);
    mPrevPhaseDeltaT = ArrayXd::Zero(mBins);
    mBinIndices = ArrayXd::LinSpaced(mBins, 0, mBins - 1);
  }

  void processFrame(RealVectorView in, ComplexVectorView out, index winSize,
                    index fftSize, index hopSize, double tolerance)
  {
    using namespace Eigen;
    using namespace _impl;
    using namespace std;
    using namespace std::complex_literals;
    double  gamma = 0.25645 * pow(winSize, 2); // assumes Hann window
    ArrayXd mag = asEigen<Array>(in);
    ArrayXd logMag = mag.max(epsilon).log();
    ArrayXd futureLogMag = logMag;
    ArrayXd currentLogMag = mPrevLogMag;
    ArrayXd prevLogMag = mPrevPrevLogMag;

    ArrayXd phaseDeltaT =
        getPhaseDeltaT(currentLogMag, gamma, fftSize, hopSize);
    ArrayXd phaseDeltaF =
        getPhaseDeltaF(prevLogMag, futureLogMag, gamma, fftSize, hopSize);
    double absTol =
        log(tolerance) + max(currentLogMag.maxCoeff(), prevLogMag.maxCoeff());
    ArrayXd todo = (currentLogMag > absTol).cast<double>();
    index   numTodo = todo.sum();
    ArrayXd phaseEst = pi + ArrayXd::Random(mBins) * pi;

    vector<pair<double, index>> heap;

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

    ArrayXd finalPhase =
        phaseEst - ArrayXd::LinSpaced(mBins, 0, 1) * pi * (winSize - 1) / 2;
    ArrayXcd result = mPrevMag * (1i * finalPhase).exp();
    mPrevPrevLogMag = mPrevLogMag;
    mPrevLogMag = logMag;
    mPrevPhase = phaseEst;
    mPrevPhaseDeltaT = phaseDeltaT;
    mPrevMag = mag;
    out <<= asFluid(result);
  }

private:
  Eigen::ArrayXd getPhaseDeltaT(Eigen::Ref<ArrayXd> logMag, double gamma,
                                index fftSize, index hopSize)
  {
    ArrayXd deltaT = ArrayXd::Zero(mBins);
    deltaT.segment(1, mBins - 2) =
        logMag.segment(2, mBins - 2) - logMag.segment(0, mBins - 2);
    deltaT = deltaT * 0.5 * hopSize * fftSize / gamma;
    deltaT = deltaT + twoPi * hopSize * mBinIndices / fftSize;
    return deltaT;
  }

  Eigen::ArrayXd getPhaseDeltaF(ArrayXd prevLogMag, ArrayXd nextLogMag,
                                double gamma, index fftSize, index hopSize)
  {
    return 0.5 * (nextLogMag - prevLogMag) * (-gamma / (hopSize * fftSize));
  }

  index   mBins;
  ArrayXd mBinIndices;
  ArrayXd mPrevMag;
  ArrayXd mPrevLogMag;
  ArrayXd mPrevPrevLogMag;
  ArrayXd mPrevPhase;
  ArrayXd mPrevPhaseDeltaT;
  ArrayXd mPrevPrevPhase;
};
} // namespace algorithm
} // namespace fluid
