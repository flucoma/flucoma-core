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

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <map>

namespace fluid {
namespace algorithm {

class NNActivations {

public:

  enum class Activation {
    kLinear,
    kSigmoid,
    kReLU,
    kTanh
  };

  using ArrayXXd = Eigen::ArrayXXd;
  using ActivationsMap =
      std::map<Activation,
      std::function<void( Eigen::Ref<Eigen::ArrayXXd>, Eigen::Ref<Eigen::ArrayXXd>)>>;

  static ActivationsMap &activation() {
    static ActivationsMap _funcs = {
      {Activation::kLinear,
      [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
          out = in; }},
        {Activation::kSigmoid,
        [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
            out = 1 / (1 + (-in).exp()); }},
        {Activation::kReLU,
        [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
            out = in.max(0); }},
         {Activation::kTanh,
        [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
             out = (in.exp() - (-in).exp()) / (in.exp() + (-in).exp()); }},
    };
    return _funcs;
  }

  static ActivationsMap &derivative() {
    static ActivationsMap _funcs = {
       {Activation::kLinear,
        [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
         out = ArrayXXd::Ones(in.rows(), in.cols());
        }},
        {Activation::kSigmoid,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
           ArrayXXd sigmoid = ArrayXXd::Zero(in.rows(), in.cols());
           activation()[Activation::kSigmoid](in, sigmoid);
           out = sigmoid * (1 - sigmoid);
         }},
        {Activation::kReLU,
         [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out){
           out = (in > 0).cast<double>();
         }},
         {Activation::kTanh,
          [](Eigen::Ref<Eigen::ArrayXXd> in, Eigen::Ref<Eigen::ArrayXXd> out) {
            ArrayXXd tanh = ArrayXXd::Zero(in.rows(), in.cols());
            activation()[Activation::kTanh](in, tanh);
            out = 1 - tanh.square();
          }}
    };
    return _funcs;
  }
};
} // namespace algorithm
} // namespace fluid
