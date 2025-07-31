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

#include "TransientExtraction.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <cmath>

namespace fluid {
namespace algorithm {

class TransientSegmentation : public algorithm::TransientExtraction
{

public:
  using TransientExtraction::TransientExtraction;

  void setDetectionParameters(double power, double threshHi, double threshLo,
      index halfWindow = 7, index hold = 25, index minSegment = 50)
  {
    TransientExtraction::setDetectionParameters(
        power, threshHi, threshLo, halfWindow, hold);
    mMinSegment = minSegment;
  }

  void init(index order, index blockSize, index padSize)
  {
    TransientExtraction::init(order, blockSize, padSize);
    mLastDetection = false;
    mDebounce = 0;
  }

  void process(const RealVectorView input, RealVectorView output,
      Allocator& alloc = FluidDefaultAllocator())
  {
    detect(input.data(), input.extent(0), alloc);
    const double* transientDetection = getDetect();
    for (index i = 0; i < std::min<index>(hopSize(), output.size()); i++)
    {
      output(i) = (transientDetection[i] != 0 && !mLastDetection && !mDebounce);
      mDebounce =
          output(i) == 1.0 ? mMinSegment : std::max<index>(0, mDebounce - 1);
      mLastDetection = transientDetection[i] == 1.0;
    }
  }

private:
  index mMinSegment{25};
  index mDebounce{0};
  bool  mLastDetection{false};
};

} // namespace algorithm
} // namespace fluid
