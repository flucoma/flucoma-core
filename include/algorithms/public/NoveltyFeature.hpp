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

#include "../util/FluidEigenMappings.hpp"
#include "../util/Novelty.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class NoveltyFeature
{

public:
  using ArrayXd = Eigen::ArrayXd;

  NoveltyFeature(index maxKernelSize, index maxFilterSize)
      : mFilterBufferStorage(maxFilterSize), mNovelty(maxKernelSize)
  {}

  void init(index kernelSize, index filterSize, index nDims)
  {
    assert(kernelSize % 2);
    mNovelty.init(kernelSize, nDims);
    mFilterBuffer = mFilterBufferStorage.segment(0, filterSize);
    mFilterBuffer.setZero();
  }

  double processFrame(const RealVectorView input)
  {
    double novelty = mNovelty.processFrame(_impl::asEigen<Eigen::Array>(input));
    index  filterSize = mFilterBuffer.size();

    if (filterSize > 1)
    {
      mFilterBuffer.segment(0, filterSize - 1) =
          mFilterBuffer.segment(1, filterSize - 1);
    }

    mFilterBuffer(filterSize - 1) = novelty;

    return mFilterBuffer.mean();
  }

private:
  ArrayXd mFilterBuffer;
  ArrayXd mFilterBufferStorage;
  Novelty mNovelty;
};
} // namespace algorithm
} // namespace fluid
