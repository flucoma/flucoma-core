/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "BufferAdaptor.hpp"
#include "OfflineClient.hpp"
#include "Result.hpp"
#include "../../data/FluidTensor.hpp"
#include <algorithm>
#include <vector>

namespace fluid {
namespace client {
namespace impl {

template <typename T>
Result spikesToTimes(FluidTensorView<T, 2> changePoints, BufferAdaptor* output,
                     size_t hopSize, size_t timeOffset, size_t /*numFrames*/,
                     double sampleRate)
{

  std::vector<size_t> numSpikes(changePoints.rows());

  for (auto i = 0; i < changePoints.rows(); ++i)
    numSpikes[i] = std::accumulate(changePoints.row(i).begin(),
                                   changePoints.row(i).end(), 0);

  // if the number of spikes doesn't match, that's a badness, and warrants an
  // abort
  assert(std::all_of(numSpikes.begin(), numSpikes.end(),
                     [&numSpikes](int a) { return a == numSpikes[0]; }));

  if (numSpikes[0] == 0)
  {
    auto idx = BufferAdaptor::Access(output);
    idx.resize(1, changePoints.rows(), sampleRate);
    double result = -1.0;
    for (auto i = 0; i < changePoints.rows(); i++) idx.samps(i)[0] = result;
    return {};
  }

  auto   idx = BufferAdaptor::Access(output);
  Result resizeResult =
      idx.resize(numSpikes[0], changePoints.rows(), sampleRate);
  if (!resizeResult.ok()) return resizeResult;

  for (auto i = 0; i < changePoints.rows(); ++i)
  {
    // Arg sort
    std::vector<size_t> indices(changePoints.row(i).size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i1, size_t i2) {
      return changePoints.row(i)[i1] > changePoints.row(i)[i2];
    });

    // Now put the gathered indicies into ascending order
    std::sort(indices.begin(), indices.begin() + numSpikes[0]);

    // convert frame numbers to samples, and account for any offset
    std::transform(indices.begin(), indices.begin() + numSpikes[0],
                   indices.begin(), [&](size_t x) -> size_t {
                     x *= hopSize;
                     x += timeOffset;
                     return x;
                   });

    idx.samps(i) = FluidTensorView<size_t, 1>{indices.data(), 0, numSpikes[0]};
  }
  return {};
}
} // namespace impl
} // namespace client
} // namespace fluid
