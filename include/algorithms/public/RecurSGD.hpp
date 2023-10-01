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

  using InputDataSeriesView = const rt::vector<InputRealMatrixView>;

public:
  explicit RecurSGD() = default;
  ~RecurSGD() = default;

  double trainManyToOne(Recur<CellType>& model, InputDataSeriesView in,
                        InputRealMatrixView out, index nIter, index batchSize,
                        double learningRate, double maxCost = 100)
  {
    assert(model.outputDims() == out.cols());
    assert(in.end() == std::find_if(in.begin(), in.end(), [&model](auto& x) {
             return model.inputDims() != x.cols();
           }));

    rt::vector<index> permutation(in.size());
    std::iota(permutation.begin(), permutation.end(), 0);

    double error;
    index  normalise;

    while (nIter-- > 0)
    {
      error = 0.0;
      normalise = 0;

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
        normalise += thisBatchSize;
      }

      if (error > maxCost) model.clear();
    }

    model.setTrained(true);
    return error / normalise;
  }

  double trainManyToMany(Recur<CellType>& model, InputDataSeriesView in,
                         InputDataSeriesView out, index nIter, index batchSize,
                         double learningRate, double maxCost = 100)
  {
    assert(in.end() == std::find_if(in.begin(), in.end(), [&model](auto& x) {
             return model.inputDims() != x.cols();
           }));
    assert(out.end() == std::find_if(out.begin(), out.end(), [&model](auto& x) {
             return model.outputDims() != x.cols();
           }));

    rt::vector<index> permutation(in.size());
    std::iota(permutation.begin(), permutation.end(), 0);

    double error;
    index  normalise;

    while (nIter-- > 0)
    {
      error = 0.0;
      normalise = 0;

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
        normalise += thisBatchSize;
      }

      if (error > maxCost) model.clear();
    }

    model.setTrained(true);
    return error / normalise;
  }


  double trainPredictor(Recur<CellType>& model, InputDataSeriesView data,
                        index nIter, index batchSize, double learningRate,
                        double maxCost = 100)
  {
    assert(model.inputDims() == model.outputDims());
    assert(data.end() ==
           std::find_if(data.begin(), data.end(), [&model](auto& x) {
             return model.dims() != x.cols();
           }));

    rt::vector<index> permutation(data.size());
    std::iota(permutation.begin(), permutation.end(), 0);

    double error;
    index  normalise;

    while (nIter-- > 0)
    {
      error = 0.0;
      normalise = 0;

      std::shuffle(permutation.begin(), permutation.end(),
                   std::mt19937{std::random_device{}()});

      for (index batchStart = 0; batchStart < data.size();
           batchStart += batchSize)
      {
        index thisBatchSize = (batchStart + batchSize) < data.size()
                                  ? batchSize
                                  : data.size() - batchStart;

        model.reset();
        for (index i = batchStart; i < batchStart + thisBatchSize; ++i)
          error += model.fit(data[permutation[i]]);

        model.update(learningRate);
        normalise += thisBatchSize;
      }

      if (error > maxCost) model.clear();
    }

    model.setTrained(true);
    return error / normalise;
  }
};

} // namespace algorithm
} // namespace fluid