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

#include "../util/FluidEigenMappings.hpp"
#include "../util/Novelty.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class NoveltyFeature
{

public:
  using ArrayXd = Eigen::ArrayXd;

  NoveltyFeature(index maxKernelSize, index maxDims, index maxFilterSize,
                 Allocator& alloc)
      : mFilterBuffer(maxFilterSize, alloc),
        mNovelty(maxKernelSize, maxDims, alloc)
  {}

  void init(index kernelSize, index filterSize, index nDims, Allocator& alloc)
  {
    assert(kernelSize % 2);
    mNovelty.init(kernelSize, nDims, alloc);
    mFilterBuffer.head(filterSize).setZero();
    mFilterSize = filterSize;
    mInitialized = true;
  }

  double processFrame(const RealVectorView input, Allocator& alloc)
  {
    assert(mInitialized);
    double novelty =
        mNovelty.processFrame(_impl::asEigen<Eigen::Array>(input), alloc);

    if (mFilterSize > 1)
    {
      mFilterBuffer.segment(0, mFilterSize - 1) =
          mFilterBuffer.segment(1, mFilterSize - 1);
    }

    mFilterBuffer(mFilterSize - 1) = novelty;

    return mFilterBuffer.head(mFilterSize).mean();
  }

private:
  bool                    mInitialized;
  ScopedEigenMap<ArrayXd> mFilterBuffer;
  Novelty                 mNovelty;
  index                   mFilterSize;
};
} // namespace algorithm
} // namespace fluid
