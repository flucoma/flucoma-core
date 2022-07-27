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

#include "AlgorithmUtils.hpp"
#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <map>

namespace fluid {
namespace algorithm {

class OnsetDetectionFuncs
{

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

  using ArrayXcd = Eigen::ArrayXcd;
  using ArrayXd = Eigen::ArrayXd;
  using ODFMap =
      std::map<ODF, std::function<double(Eigen::Ref<ArrayXcd>, Eigen::Ref<ArrayXcd>, Eigen::Ref<ArrayXcd>)>>;

  static ArrayXd wrapPhase(ArrayXd phase)
  {
    return phase.unaryExpr([=](const double p) {
      return p > (-pi) && p > pi
                 ? p
                 : p + (twoPi) * (1.0 + floor((-pi - p) / twoPi));
    });
  }

  static ODFMap& map()
  {
    static ODFMap _funcs = {

        {ODF::kEnergy,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> /*prev*/, Eigen::Ref<ArrayXcd> /*prevprev*/) {
           return cur.abs().real().square().mean();
         }},
        {ODF::kHFC,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> /*prev*/, Eigen::Ref<ArrayXcd> /*prevprev*/) {
           index   n = cur.size();
           ArrayXd space = ArrayXd(n);
           space.setLinSpaced(0, n);
           return (space * cur.abs().real().square()).mean();
         }},
        {ODF::kSpectralFlux,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> prev, Eigen::Ref<ArrayXcd> /*prevprev*/) {
           return (cur.abs().real() - prev.abs().real()).max(0.0).mean();
         }},
        {ODF::kMKL,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> prev, Eigen::Ref<ArrayXcd> /*prevprev*/) {
           ArrayXd mag1 = cur.abs().real().max(epsilon);
           ArrayXd mag2 = prev.abs().real().max(epsilon);
           return (mag1 / mag2).max(epsilon).log().mean();
         }},
        {ODF::kIS,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> prev, Eigen::Ref<ArrayXcd> /*prevprev*/) {
           ArrayXd mag1 = cur.abs().real().max(epsilon);
           ArrayXd mag2 = prev.abs().real().max(epsilon);
           ArrayXd ratio = (mag1 / mag2).square().max(epsilon);
           return (ratio - ratio.log() - 1).mean();
         }},
        {ODF::kCosine,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> prev, Eigen::Ref<ArrayXcd> /*prevprev*/) {
           ArrayXd mag1 = cur.abs().real().max(epsilon);
           ArrayXd mag2 = prev.abs().real().max(epsilon);
           double  norm = mag1.matrix().norm() * mag2.matrix().norm();
           double  dot = mag1.matrix().dot(mag2.matrix());
           return 1 - dot / norm;
         }},
        {ODF::kPhaseDev,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> prev, Eigen::Ref<ArrayXcd> prevprev) {
           ArrayXd phaseAcc = (cur.atan().real() - prev.atan().real()) -
                              (prev.atan().real() - prevprev.atan().real());
           return wrapPhase(phaseAcc).mean();
         }},
        {ODF::kWPhaseDev,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> prev, Eigen::Ref<ArrayXcd> prevprev) {
           ArrayXd mag1 = cur.abs().real().max(epsilon);
           ArrayXd phaseAcc = (cur.atan().real() - prev.atan().real()) -
                              (prev.atan().real() - prevprev.atan().real());
           return wrapPhase(mag1 * phaseAcc).mean();
         }},
        {ODF::kComplexDev,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> prev, Eigen::Ref<ArrayXcd> prevprev) {
           ArrayXcd target(cur.size());
           ArrayXd  prevMag = prev.abs().real().max(epsilon);
           ArrayXd  prevPhase = prev.atan().real();
           ArrayXd  phaseEst = wrapPhase(
               prevPhase + (prev.atan().real() - prevprev.atan().real()));
           target.real() = prevMag * phaseEst.cos();
           target.imag() = prevMag * phaseEst.sin();
           return (target - cur).abs().real().mean();
         }},
        {ODF::kRComplexDev,
         [](Eigen::Ref<ArrayXcd> cur, Eigen::Ref<ArrayXcd> prev, Eigen::Ref<ArrayXcd> prevprev) {
           ArrayXcd target(cur.size());
           ArrayXd  prevMag = prev.abs().real().max(epsilon);
           ArrayXd  prevPhase = prev.atan().real();
           ArrayXd  phaseEst = wrapPhase(
               prevPhase + (prev.atan().real() - prevprev.atan().real()));
           target.real() = prevMag * phaseEst.cos();
           target.imag() = prevMag * phaseEst.sin();
           return (target - cur).abs().real().max(0.0).mean();
         }},
    };
    return _funcs;
  }
};
} // namespace algorithm
} // namespace fluid
