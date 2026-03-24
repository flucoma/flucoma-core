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

#include "FluidIndex.hpp"
#include "detail/DataSampler.hpp"
#include <algorithm>
#include <random>
#include <vector>


namespace fluid {

class SimpleDataSampler : public detail::DataSampler<SimpleDataSampler>
{
  friend detail::DataSampler<SimpleDataSampler>;

  template <class InputIter>
  FluidTensorView<index, 2> map(InputIter start, InputIter end,
                                FluidTensorView<index, 2> dst)
  {
    using std::begin, std::copy;
    auto inputSamples = dst.col(0);
    auto outputSamples = dst.col(1);
    copy(start, end, begin(inputSamples));
    copy(start, end, begin(outputSamples));
    return dst;
  }

public:
  SimpleDataSampler(index size, index batchSize, double validationFraction,
                    bool shuffle, index seed)
      : detail::DataSampler<SimpleDataSampler>(size, batchSize,
                                               validationFraction, shuffle, seed)
  {}
};

} // namespace fluid