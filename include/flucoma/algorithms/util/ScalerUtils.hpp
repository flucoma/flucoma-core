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

#include <Eigen/Core>
#include "../util/AlgorithmUtils.hpp"

// In scalers, the range cannot be too small otherwise it gets unmanageable as denominator
// To sanitize, we set an arbitrary threshold of 10*epsilon and replace by 1 if smaller
// This is in line with scikit learn behaviour (https://github.com/scikit-learn/scikit-learn/blob/16625450b58f555dc3955d223f0c3b64a5686984/sklearn/preprocessing/_data.py#L88-L118)

void handleZerosInScale(Eigen::ArrayXd& rangeArray)
{
  rangeArray = (rangeArray < 10 * fluid::algorithm::epsilon).select(1,rangeArray);
}

void handleZerosInScale(double& range)
{
  range = (range < (10 * fluid::algorithm::epsilon)) ? 1 : range;
}
