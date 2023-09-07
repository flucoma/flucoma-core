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
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <random>

namespace fluid {
namespace algorithm {

template <class CellType>
class RecurSGD
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

public:
  explicit RecurSGD() = default;
  ~RecurSGD() = default;

  double trainManyToOne(Recur<CellType>&                      model,
                        const rt::vector<InputRealMatrixView> in,
                        InputRealMatrixView out, index nIter, index batchSize,
                        double learningRate)
  {
    assert(model.dims() == in[0].cols());
    assert(model.size() == out.cols());
    assert(std::adjacent_find(in.begin(), in.end(), [](auto& a, auto& b) {
             return a.cols() != b.cols();
           }) == in.end());

    rt::vector<index> permutation(in.size());
    std::iota(permutation.begin(), permutation.end(), 0);

    double error;
    index  nBatches;

    while (nIter-- > 0)
    {
      error = 0.0;
      nBatches = 0;
      std::shuffle(permutation.begin(), permutation.end(),
                   std::mt19937{std::random_device{}()});

      for (index batchStart = 0; asUnsigned(batchStart) < in.size();
           batchStart += batchSize)
      {
        index thisBatchSize = asUnsigned(batchStart + batchSize) < in.size()
                                  ? batchSize
                                  : in.size() - batchStart;

        model.reset();
        for (index i = batchStart; i < batchStart + thisBatchSize; ++i)
          error += model.fit(in[permutation[i]], out.row(permutation[i]));

        model.update(learningRate);
        ++nBatches;
      }
    }

    model.setTrained();
    return error / nBatches;
  }

  double trainManyToMany(Recur<CellType>&                      model,
                         const rt::vector<InputRealMatrixView> in,
                         const rt::vector<InputRealMatrixView> out, index nIter,
                         index batchSize, double learningRate)
  {
    assert(model.dims() == in[0].cols());
    assert(model.size() == out[0].cols());
    assert(std::adjacent_find(in.begin(), in.end(), [](auto& a, auto& b) {
             return a.cols() != b.cols();
           }) == in.end());
    assert(std::adjacent_find(out.begin(), out.end(), [](auto& a, auto& b) {
             return a.cols() != b.cols();
           }) == out.end());

    rt::vector<index> permutation(in.size());
    std::iota(permutation.begin(), permutation.end(), 0);

    double error;
    index  nBatches;
    while (nIter-- > 0)
    {
      error = 0.0;
      nBatches = 0;
      std::shuffle(permutation.begin(), permutation.end(),
                   std::mt19937{std::random_device{}()});

      for (index batchStart = 0; batchStart < in.size();
           batchStart += batchSize)
      {
        index thisBatchSize = (batchStart + batchSize) < in.size()
                                  ? batchSize
                                  : in.size() - batchStart;

        model.reset();
        for (index i = batchStart; i < batchStart + thisBatchSize; ++i)
          error += model.fit(in[permutation[i]], out[permutation[i]]);

        model.update(learningRate);
        ++nBatches;
      }
    }

    model.setTrained();
    return error / nBatches;
  }
};

} // namespace algorithm
} // namespace fluid