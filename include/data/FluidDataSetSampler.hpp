/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

/*
Pairing two unordered DataSets for supervised use
*/
#pragma once

#include "FluidDataSet.hpp"
#include "FluidIndex.hpp"
#include "detail/DataSampler.hpp"
#include <optional>
#include <random>


namespace fluid {

class FluidDataSetSampler
    : public detail::DataSampler<FluidDataSetSampler>
{
  friend detail::DataSampler<FluidDataSetSampler>;

  std::pair<std::vector<index>, std::vector<index>> mIdxMaps;

  template <class InputIter>
  FluidTensorView<index, 2> map(InputIter start, InputIter end,
                                FluidTensorView<index, 2> dst)
  {
    using std::begin;
    auto inputSamples = dst.col(0);
    auto outputSamples = dst.col(1);

    transform(start, end, begin(inputSamples),
              [&idx = mIdxMaps](index i) { return idx.first[i]; });
    transform(start, end, begin(outputSamples),
              [&idx = mIdxMaps](index i) { return idx.second[i]; });
    return dst;
  }

public:
  template <typename DataSetA, typename DataSetB>
  FluidDataSetSampler(DataSetA const& in, DataSetB const& out, index batchSize,
                      double validationFraction, bool shuffle = true)
      : detail::DataSampler<FluidDataSetSampler>(in.size(), batchSize,
                                                 validationFraction, shuffle),
        mIdxMaps{indexMap(in, out)}
  {}
};
} // namespace fluid