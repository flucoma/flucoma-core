#pragma once

#include <clients/common/BufferAdaptor.hpp>
#include <clients/common/OfflineClient.hpp>
#include <data/FluidTensor.hpp>
#include <algorithm>
#include <vector>

namespace fluid {
namespace client{
namespace impl{

  template<typename T>
  void spikesToTimes(FluidTensorView<T,1> changePoints, BufferAdaptor* output, size_t hopSize, size_t timeOffset, size_t numFrames, double sampleRate)
  {
    size_t numSpikes =
        std::accumulate(changePoints.begin(), changePoints.end(), 0);

    // Arg sort
    std::vector<size_t> indices(changePoints.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i1, size_t i2) {
      return changePoints[i1] > changePoints[i2];
    });

    // Now put the gathered indicies into ascending order
    std::sort(indices.begin(), indices.begin() + numSpikes);

    // convert frame numbers to samples, and account for any offset
    std::transform(indices.begin(), indices.begin() + numSpikes,
                   indices.begin(), [&](size_t x) -> size_t {
                     x *= hopSize;
                     x += timeOffset;
                     return x;
                   });

    // Place a leading <offset> and <numframes>
    
    
    indices.insert(indices.begin() + numSpikes, timeOffset + numFrames);
    int extraSpikes = 1;
    if(indices[0] > timeOffset)
    {
      indices.insert(indices.begin(), timeOffset);
      ++extraSpikes;
    }

    auto idx = BufferAdaptor::Access(output); 

    idx.resize(numSpikes + extraSpikes, 1,sampleRate);

    idx.samps(0) = FluidTensorView<size_t, 1>{indices.data(), 0, numSpikes + extraSpikes};
  }
}
}
}

