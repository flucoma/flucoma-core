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
                        double learningRate, double validation = 0.1)
  {
    assert(in.size() == out.rows());
    assert(model.outputDims() == out.cols());
    assert(in.end() == std::find_if(in.begin(), in.end(), [&model](auto& x) {
             return model.inputDims() != x.cols();
           }));

    double error = trainSGD(
        model, in.size(), nIter, batchSize, learningRate,
        validation, [&in, &out, &model](index i) -> double {
          assert(i >= 0 && i < in.size());
          assert(i >= 0 && i < out.rows());

          return model.fit(in[i], out.row(i));
        });

    return error;
  }

  double trainManyToMany(Recur<CellType>& model, InputDataSeriesView in,
                         InputDataSeriesView out, index nIter, index batchSize,
                         double learningRate, double validation = 0.1)
  {
    assert(in.size() == out.size());
    assert(in.end() == std::find_if(in.begin(), in.end(), [&model](auto& x) {
             return model.inputDims() != x.cols();
           }));
    assert(out.end() == std::find_if(out.begin(), out.end(), [&model](auto& x) {
             return model.outputDims() != x.cols();
           }));

    double error = trainSGD(model, in.size(), nIter, batchSize, learningRate,
                            validation, [&in, &out, &model](index i) -> double {
                              assert(i >= 0 && i < in.size());
                              assert(i >= 0 && i < out.size());

                              return model.fit(in[i], out[i]);
                            });

    return error;
  }


  double trainPredictor(Recur<CellType>& model, InputDataSeriesView data,
                        index nIter, index batchSize, double learningRate,
                        double validation = 0.1)
  {
    assert(model.inputDims() == model.outputDims());
    assert(data.end() ==
           std::find_if(data.begin(), data.end(), [&model](auto& x) {
             return model.dims() != x.cols();
           }));

    double error = trainSGD(model, data.size(), nIter, batchSize, learningRate,
                            validation, [&data, &model](index i) -> double {
                              assert(i >= 0 && i < data.size());
                              return model.fit(data[i]);
                            });

    return error;
  }

private:
  double trainSGD(Recur<CellType>& model, index corpusSize, index nIter,
                  index batchSize, double learningRate, double validation,
                  std::function<double(index)> fit)
  {
    rt::vector<index> permutation(corpusSize);
    std::iota(permutation.begin(), permutation.end(), 0);

    double error = 0.0, prevError = 0.0;

    index patience = 10;
    index nValidate = validation * corpusSize;
    index nTrain = corpusSize - nValidate;

    while (nIter-- > 0 && patience > 0)
    {
      error = 0.0;

      std::shuffle(permutation.begin(), permutation.end(),
                   std::mt19937{std::random_device{}()});

      for (index batchStart = 0; batchStart < nTrain;
           batchStart += batchSize)
      {
        index thisBatchSize = (batchStart + batchSize) < nTrain
                                  ? batchSize
                                  : nTrain - batchStart;

        model.reset();
        for (index i = batchStart; i < batchStart + thisBatchSize; ++i)
          error += fit(permutation[i]);

        model.update(learningRate);
      }

      for (index i = nTrain; i < corpusSize; ++i)
      {
        error += fit(permutation[i]);
      }

      if (error > prevError)
      {
        prevError = error;
        --patience;
      }
    }

    model.setTrained(true);

    for (index i = 0; i < corpusSize; ++i)
    {
      error += fit(i);
    }

    return error / corpusSize;
  }
};

} // namespace algorithm
} // namespace fluid