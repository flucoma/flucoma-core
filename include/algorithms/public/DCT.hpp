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
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class DCT
{
public:
  using ArrayXd = Eigen::ArrayXd;
  using MatrixXd = Eigen::MatrixXd;

  DCT(index maxInputSize, index maxOutputSize,
      Allocator& alloc = FluidDefaultAllocator())
      : mTable{maxOutputSize, maxInputSize, alloc}
  {}

  void init(index inputSize, index outputSize,
      Allocator& alloc = FluidDefaultAllocator())
  {
    using namespace std;
    assert(inputSize >= outputSize);
    assert(inputSize <= mTable.cols());
    assert(outputSize <= mTable.rows());

    // Do not reinitialise if there is no need
      
    if (mInitialized && mInputSize == inputSize && mOutputSize == outputSize)
      return;
      
    mInputSize = inputSize;
    mOutputSize = outputSize;
    mTable.setZero();

    ScopedEigenMap<ArrayXd> freqs(inputSize, alloc);
    for (index i = 0; i < mOutputSize; i++)
    {
      double scale = i == 0 ? 1.0 / sqrt(inputSize) : sqrt(2.0 / inputSize);
      freqs = ((pi / inputSize) * i) *
              ArrayXd::LinSpaced(inputSize, 0.5, inputSize - 0.5);
      mTable.topLeftCorner(outputSize, inputSize).row(i) = freqs.cos() * scale;
    }
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out)
  {
    assert(mInitialized && "DCT: processFrame() called before init()");
    assert(in.size() == mInputSize &&
           "DCT: actual input size doesn't match expected size");
    assert(out.size() == mOutputSize &&
           "DCT: actual output size doesn't maatch expected size");

    FluidEigenMap<Eigen::Matrix> frame = _impl::asEigen<Eigen::Matrix>(in);
    _impl::asEigen<Eigen::Matrix>(out).noalias() =
        (mTable.topLeftCorner(mOutputSize, mInputSize) * frame);
  }

  void processFrame(Eigen::Ref<const ArrayXd> input, Eigen::Ref<ArrayXd> output)
  {
    output.matrix().noalias() =
        (mTable.topLeftCorner(mOutputSize, mInputSize) * input.matrix());
  }

private:
  index                    mInputSize{40};
  index                    mOutputSize{13};
  bool                     mInitialized{false};
  ScopedEigenMap<MatrixXd> mTable;
};
} // namespace algorithm
} // namespace fluid
