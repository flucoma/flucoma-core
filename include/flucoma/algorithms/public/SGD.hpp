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

#include "MLP.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/SimpleDataSampler.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <limits>

namespace fluid {
namespace algorithm {

class SGD
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

public:

  double train(MLP& model, InputRealMatrixView in, RealMatrixView out,
               index nIter, index batchSize, double learningRate,
               double momentum, double valFrac)
  {
    return train(model, in, out,
                 SimpleDataSampler(in.rows(), batchSize, valFrac, true), nIter,
                 learningRate, momentum);
  }

  template <typename Sampler>
  double train(MLP& model, InputRealMatrixView in, RealMatrixView out,
               Sampler&& loader, index nIter, double learningRate,
               double momentum)
  {
    using namespace _impl;
    using namespace std;
    using namespace Eigen;
    index nExamples = in.rows();
    // index inputSize = in.cols();
    index outputSize = out.cols();

    auto                    valIdx = loader.validationSet();
    std::optional<ArrayXXd> valInput;
    std::optional<ArrayXXd> valOutput;
    if (valIdx)
    {
      valInput =
          ArrayXXd(asEigen<Eigen::Array>(in)(valIdx->col(0), Eigen::all));
      valOutput =
          ArrayXXd(asEigen<Eigen::Array>(out)(valIdx->col(1), Eigen::all));
    }

    double error = 0;
    index  patience = mInitialPatience;
    double prevValLoss = std::numeric_limits<double>::max();
    while (nIter-- > 0)
    {
      for (auto batch : loader)
      {
        index    thisBatchSize = batch->rows();
        ArrayXXd batchIn = asEigen<Eigen::Array>(in)(batch->col(0), Eigen::all);
        ArrayXXd batchOut =
            asEigen<Eigen::Array>(out)(batch->col(1), Eigen::all);
        ArrayXXd batchPred = ArrayXXd::Zero(thisBatchSize, outputSize);
        model.forward(batchIn, batchPred);
        ArrayXXd diff = batchPred - batchOut;
        model.backward(diff);
        model.update(learningRate, momentum);
      }
      if (valIdx)
      {
        ArrayXXd valPred = ArrayXXd::Zero(valInput->rows(), outputSize);
        model.forward(*valInput, valPred);
        double valLoss = model.loss(valPred, *valOutput);
        if (valLoss < prevValLoss)
          patience = mInitialPatience;
        else
          patience--;
        if (patience <= 0) break;
        prevValLoss = valLoss;
      }
    }

    auto trainingIdx = loader.trainingSet();
    nExamples = trainingIdx->rows();
    ArrayXXd input = asEigen<Eigen::Array>(in)(trainingIdx->col(0), Eigen::all);
    ArrayXXd output =
        asEigen<Eigen::Array>(out)(trainingIdx->col(1), Eigen::all);

    ArrayXXd finalPred = ArrayXXd::Zero(nExamples, outputSize);
    model.forward(input, finalPred);
    bool isNan = !((finalPred == finalPred)).all();
    if (isNan)
    {
      model.clear(-1);//this is wrong OWEN?
      return -1;
    }
    error = model.loss(finalPred, output);
    model.setTrained(true);
    return error;
  }

private:
  index mInitialPatience{10};
};
} // namespace algorithm
} // namespace fluid
