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

#include "AlgorithmUtils.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <map>

namespace fluid {
namespace algorithm {

class OnsetDetectionFuncs
{

  using ArrayXcd = Eigen::ArrayXcd;
  using ArrayXd = Eigen::ArrayXd;
  using Ref = Eigen::Ref<ArrayXcd>;

  static auto wrapPhase()
  {
    return [=](const double p) {
      return p > (-pi) && p > pi
                 ? p
                 : p + (twoPi) * (1.0 + floor((-pi - p) / twoPi));
    };
  }

  static double energy(Ref cur, Ref /*prev*/, Ref /*prevprev*/, Allocator&)
  {
    return cur.abs().real().square().mean();
  }

  static double HFC(Ref cur, Ref /*prev*/, Ref /*prevprev*/, Allocator&)
  {
    index n = cur.size();
    auto  space = ArrayXd::LinSpaced(n, 0, n);
    return (space * cur.abs().real().square()).mean();
  }

  static double spectralFlux(Ref cur, Ref prev, Ref /*prevprev*/, Allocator&)
  {
    return (cur.abs().real() - prev.abs().real()).max(0.0).mean();
  }

  static double MKL(Ref cur, Ref prev, Ref /*prevprev*/, Allocator&)
  {
    auto mag1 = cur.abs().real().max(epsilon);
    auto mag2 = prev.abs().real().max(epsilon);
    return (mag1 / mag2).max(epsilon).log().mean();
  }

  static double IS(Ref cur, Ref prev, Ref /*prevprev*/, Allocator& alloc)
  {
    auto                    mag1 = cur.abs().real().max(epsilon);
    auto                    mag2 = prev.abs().real().max(epsilon);
    ScopedEigenMap<ArrayXd> ratio(cur.size(), alloc);
    ratio = (mag1 / mag2).square().max(epsilon);
    return (ratio - ratio.log() - 1).mean();
  }

  static double cosine(Ref cur, Ref prev, Ref /*prevprev*/, Allocator& alloc)
  {
    ScopedEigenMap<ArrayXd> mag1(cur.abs().real().max(epsilon), alloc);
    ScopedEigenMap<ArrayXd> mag2(prev.abs().real().max(epsilon), alloc);
    double                  norm = mag1.matrix().norm() * mag2.matrix().norm();
    double                  dot = mag1.matrix().dot(mag2.matrix());
    return 1 - dot / norm;
  }

  static double phaseDev(Ref cur, Ref prev, Ref prevprev, Allocator& alloc)
  {
    ScopedEigenMap<ArrayXd> phaseAcc{
        (cur.atan().real() - prev.atan().real()) -
            (prev.atan().real() - prevprev.atan().real()),
        alloc};
    phaseAcc = phaseAcc.unaryExpr(wrapPhase());
    return phaseAcc.mean();
  }

  static double wPhaseDev(Ref cur, Ref prev, Ref prevprev, Allocator& alloc)
  {
    ScopedEigenMap<ArrayXd> mag1(cur.abs().real().max(epsilon), alloc);
    ScopedEigenMap<ArrayXd> phaseAcc(
        (cur.atan().real() - prev.atan().real()) -
            (prev.atan().real() - prevprev.atan().real()),
        alloc);
    phaseAcc = phaseAcc * mag1;
    phaseAcc = phaseAcc.unaryExpr(wrapPhase());
    return phaseAcc.mean();
  }

  static double complexDev(Ref cur, Ref prev, Ref prevprev, Allocator& alloc)
  {
    ScopedEigenMap<ArrayXcd> target(cur.size(), alloc);
    ScopedEigenMap<ArrayXd>  prevMag(prev.abs().real().max(epsilon), alloc);
    ScopedEigenMap<ArrayXd>  prevPhase(prev.atan().real(), alloc);
    ScopedEigenMap<ArrayXd>  phaseEst(
         prevPhase + (prev.atan().real() - prevprev.atan().real()), alloc);
    phaseEst = phaseEst.unaryExpr(wrapPhase());
    target.real() = prevMag * phaseEst.cos();
    target.imag() = prevMag * phaseEst.sin();
    return (target - cur).abs().real().mean();
  }

  static double rComplexDev(Ref cur, Ref prev, Ref prevprev, Allocator& alloc)
  {
    ScopedEigenMap<ArrayXcd> target(cur.size(), alloc);
    ScopedEigenMap<ArrayXd>  prevMag(prev.abs().real().max(epsilon), alloc);
    ScopedEigenMap<ArrayXd>  prevPhase(prev.atan().real(), alloc);
    ScopedEigenMap<ArrayXd>  phaseEst(
         prevPhase + (prev.atan().real() - prevprev.atan().real()), alloc);
    phaseEst = phaseEst.unaryExpr(wrapPhase());
    target.real() = prevMag * phaseEst.cos();
    target.imag() = prevMag * phaseEst.sin();
    return (target - cur).abs().real().max(0.0).mean();
  }

public:
  enum class ODF {
    kEnergy,
    kHFC,
    kSpectralFlux,
    kMKL,
    kIS,
    kCosine,
    kPhaseDev,
    kWPhaseDev,
    kComplexDev,
    kRComplexDev
  };

  static auto& map(index functionIndex)
  {
    using ODFTable = std::array<double (*)(Ref, Ref, Ref, Allocator&), 10>;
    static ODFTable _funcs{energy,     HFC,        spectralFlux, MKL,
                           IS,         cosine,     phaseDev,     wPhaseDev,
                           complexDev, rComplexDev};

    assert(functionIndex < asSigned(_funcs.size()));

    return _funcs[asUnsigned(functionIndex)];
  }

  static auto& map(ODF function) { return map(static_cast<index>(function)); }
};
} // namespace algorithm
} // namespace fluid
