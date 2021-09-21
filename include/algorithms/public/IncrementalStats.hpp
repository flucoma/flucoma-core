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

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/IncrementalMeanVar.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class IncrementalStats
{
public:

  void clear()
  {
    mSampleCount = 0; 
  }

  template<typename T>
  void process(FluidTensorView<T,1> in, FluidTensorView<T,1> mean, FluidTensorView<T,1> var)
  {
    index size = in.size();
    
    if (size != mInFrame.cols())
    {
      mInFrame.resize(1, size);
      mMean.resize(size);
      mVar.resize(size);
      mVar.fill(0); 
    }

    mInFrame.row(0) = _impl::asEigen<Eigen::Array>(in)
                          .transpose()
                          .template cast<double>();
    
    if(mSampleCount == 0)
    {
      mVar = mInFrame.row(0); 
    }

    mSampleCount =
        _impl::incrementalMeanVariance(mInFrame, mSampleCount, mMean, mVar);
    mean = _impl::asFluid(mMean);
    var = _impl::asFluid(mVar);
  }

private:
  index           mSampleCount{0};
  Eigen::ArrayXXd mInFrame;
  Eigen::ArrayXd  mMean;
  Eigen::ArrayXd  mVar;
};
} // namespace algorithm
} // namespace fluid
