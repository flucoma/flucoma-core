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

#include "../util/AlgorithmUtils.hpp"
#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <map>

namespace fluid {
namespace algorithm {

class WindowFuncs
{
public:
  enum class WindowTypes {
    kHann,
    kHannD,
    kHamming,
    kBlackmanHarris,
    kGaussian
  };
  using WindowFuncsMap =
      std::map<WindowTypes,
               std::function<void(index, Eigen::Ref<Eigen::ArrayXd>)>>;

  static WindowFuncsMap& map()
  {
    using namespace std;
    static WindowFuncsMap _funcs = {
        {WindowTypes::kHann,
         [](index size, Eigen::Ref<Eigen::ArrayXd> out) {
           for (index i = 0; i < size; i++)
           { out(i) = 0.5 - 0.5 * cos((pi * 2 * i) / size); }
         }},
        {WindowTypes::kHannD,
         [](index size, Eigen::Ref<Eigen::ArrayXd> out) {
           double norm = pi / size;
           for (index i = 0; i < size; i++)
           { out(i) = norm * sin((2 * pi * i) / size); }
         }},
        {WindowTypes::kHamming,
         [](index size, Eigen::Ref<Eigen::ArrayXd> out) {
           for (index i = 0; i < size; i++)
           { out(i) = 0.54 - 0.46 * cos((pi * 2 * i) / size); }
         }},
        {WindowTypes::kBlackmanHarris,
         [](index size, Eigen::Ref<Eigen::ArrayXd> out) {
           for (index i = 0; i < size; i++)
           {
             out(i) = 0.35875 - 0.48829 * cos((pi * 2 * i) / size) +
                      0.14128 * cos((pi * 2 * i) / size) +
                      0.01168 * cos((pi * 2 * i) / size);
           }
         }},
        {WindowTypes::kGaussian,
         [](index size, Eigen::Ref<Eigen::ArrayXd> out) {
           double sigma = size / 3; // TODO: should be argument
           assert(size % 2);
           index h = (size - 1) / 2;
           for (index i = -h; i <= h; i++)
           { out(i + h) = exp(-i * i / (2 * sigma * sigma)); }
         }}};
    return _funcs;
  }
};
} // namespace algorithm
} // namespace fluid
