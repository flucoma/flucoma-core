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
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <limits>
#include <random>

namespace fluid {
namespace algorithm {

class SGD
{
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using Permutation = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

public:
  explicit SGD() = default;
  ~SGD() = default;

  double train(MLP& model, const RealMatrixView in, RealMatrixView out,
               index nIter, index batchSize, double learningRate,
               double momentum, double valFrac)
  {
    using namespace _impl;
    using namespace std;
    using namespace Eigen;
    index       nExamples = in.rows();
    index       inputSize = in.cols();
    index       outputSize = out.cols();
    ArrayXXd    input = asEigen<Eigen::Array>(in);
    ArrayXXd    output = asEigen<Eigen::Array>(out);
    Permutation valPerm(nExamples);
    valPerm.setIdentity();
    shuffle(valPerm.indices().data(),
            valPerm.indices().data() + valPerm.indices().size(),
            mt19937{random_device{}()});
    input = valPerm * input.matrix();
    output = valPerm * output.matrix();
    index nVal = std::lround(nExamples * valFrac);
    index nTrain = nExamples - nVal;

    ArrayXXd trainInput = input.block(0, 0, nTrain, inputSize);
    ArrayXXd trainOutput = output.block(0, 0, nTrain, outputSize);
    ArrayXXd valInput = input.block(nTrain, 0, nVal, inputSize);
    ArrayXXd valOutput = output.block(nTrain, 0, nVal, outputSize);

    Permutation iterPerm(nTrain);
    iterPerm.setIdentity();
    double error = 0;
    index  patience = mInitialPatience;
    double prevValLoss = std::numeric_limits<double>::max();
    while (nIter-- > 0)
    {
      shuffle(iterPerm.indices().data(),
              iterPerm.indices().data() + iterPerm.indices().size(),
              mt19937{random_device{}()});
      ArrayXXd inPerm = iterPerm * trainInput.matrix();
      ArrayXXd outPerm = iterPerm * trainOutput.matrix();
      for (index batchStart = 0; batchStart < inPerm.rows();
           batchStart += batchSize)
      {
        index thisBatchSize = (batchStart + batchSize) <= nTrain
                                  ? batchSize
                                  : nTrain - batchStart;
        ArrayXXd batchIn =
            inPerm.block(batchStart, 0, thisBatchSize, inPerm.cols());
        ArrayXXd batchOut =
            outPerm.block(batchStart, 0, thisBatchSize, outPerm.cols());
        ArrayXXd batchPred = ArrayXXd::Zero(thisBatchSize, outputSize);
        model.forward(batchIn, batchPred);
        ArrayXXd diff = batchPred - batchOut;
        model.backward(diff);
        model.update(learningRate, momentum);
      }
      if (nVal > 0)
      {
        ArrayXXd valPred = ArrayXXd::Zero(nVal, outputSize);
        model.forward(valInput, valPred);
        double valLoss = model.loss(valPred, valOutput);
        if (valLoss < prevValLoss)
          patience = mInitialPatience;
        else
          patience--;
        if (patience <= 0) break;
        prevValLoss = valLoss;
      }
    }
    ArrayXXd finalPred = ArrayXXd::Zero(nExamples, outputSize);
    model.forward(input, finalPred);
    bool isNan = !((finalPred == finalPred)).all();
    if (isNan)
    {
      model.clear();
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
