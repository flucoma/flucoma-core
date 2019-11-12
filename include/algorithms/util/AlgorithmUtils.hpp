/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include <cmath>

namespace fluid {
namespace algorithm {

constexpr double epsilon = std::numeric_limits<double>::epsilon();
constexpr double pi = M_PI;
constexpr double twoPi = 2 * M_PI;
constexpr double silence = 6.3095734448019e-08;
constexpr double silenceDB = -144;

}; // namespace algorithm
}; // namespace fluid
