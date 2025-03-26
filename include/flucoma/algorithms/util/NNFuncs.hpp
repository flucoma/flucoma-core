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

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <map>

namespace fluid {
namespace algorithm {

class NNActivations
{

public:
  enum class Activation { kLinear, kSigmoid, kReLU, kTanh };

  using ArrayXXd = Eigen::ArrayXXd;
  using ActivationsMap =
      std::map<Activation, std::function<void(Eigen::Ref<Eigen::ArrayXXd>,
                                              Eigen::Ref<Eigen::ArrayXXd>)>>;

  static ActivationsMap& activation()
  {
    static ActivationsMap _funcs = {
        {Activation::kLinear,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           out = in;
         }},
        {Activation::kSigmoid,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           out = 1 / (1 + (-in).exp());
         }},
        {Activation::kReLU,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           out = in.max(0);
         }},
        {Activation::kTanh,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           out = (in.exp() - (-in).exp()) / (in.exp() + (-in).exp());
         }},
    };
    return _funcs;
  }

  // derivative from output of activation
  static ActivationsMap& derivative()
  {
    static ActivationsMap _funcs = {
        {Activation::kLinear,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           out = ArrayXXd::Ones(in.rows(), in.cols());
         }},
        {Activation::kSigmoid,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           out = in * (1 - in);
         }},
        {Activation::kReLU,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           out = (in > 0).cast<double>();
         }},
        {Activation::kTanh,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           out = 1 - in.square();
         }}};
    return _funcs;
  }
};
} // namespace algorithm
} // namespace fluid
