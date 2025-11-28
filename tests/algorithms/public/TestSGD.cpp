#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <flucoma/algorithms/public/MLP.hpp>
#include <flucoma/algorithms/public/SGD.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <algorithm>
#include <vector>

namespace fluid::algorithm {

using Tensor = FluidTensor<double, 2>;


index  N = 64;
index  nIter = 1;
index  batchSize = N;
double learnRate = 0.1;
double momentum = 0.0;
double valFrac = 0.0;


TEST_CASE("SGD is repeatable with manually set seed")
{

  std::vector models(3, MLP());
  SGD         algo;


  Tensor input(64, 1);
  std::iota(input.begin(), input.end(), 0.0);
  Tensor output(input);

  models[0].init(1, 1, FluidTensor<index, 1>{2}, 0, 0, -1);
  models[1] = models[0];
  models[2] = models[1];

  double error = algo.train(models[0], input, output, 1, batchSize / 2,
                            learnRate, momentum, valFrac, 42);
  REQUIRE_FALSE(error == -1);
  error = algo.train(models[1], input, output, 1, batchSize / 2, learnRate,
                     momentum, valFrac, 42);
  REQUIRE_FALSE(error == -1);
  error = algo.train(models[2], input, output, 1, batchSize / 2, learnRate,
                     momentum, valFrac, 28976);
  REQUIRE_FALSE(error == -1);


  std::vector        weights(3, Tensor(1, 2));
  std::vector        biases(3, FluidTensor<double, 1>(2));
  std::vector<index> activations(3);

  models[0].getParameters(0, weights[0], biases[0], activations[0]);
  models[1].getParameters(0, weights[1], biases[1], activations[1]);
  models[2].getParameters(0, weights[2], biases[2], activations[2]);

  using Catch::Matchers::RangeEquals;
  // only weights are stochastic
  REQUIRE_THAT(weights[1], RangeEquals(weights[0]));
  REQUIRE_THAT(weights[1], !RangeEquals(weights[2]));
}

TEST_CASE("Failed training doesn't mutate model")
{

  MLP model;
  SGD algo;


  Tensor input(64, 1);
  Tensor output(64, 1);
  input.fill(0);
  // adding a NaN to fail training
  input(31, 0) = std::numeric_limits<double>::quiet_NaN();

  std::vector        weights(2, Tensor(1, 2));
  std::vector        biases(2, FluidTensor<double, 1>(2));
  std::vector<index> activations(2);

  model.init(1, 1, FluidTensor<index, 1>{2}, 0, 0, -1);

  model.getParameters(0, weights[0], biases[0], activations[0]);

  double error = algo.train(model, input, output, nIter, batchSize, learnRate,
                            momentum, valFrac, -1);
  REQUIRE(error == -1);
  model.getParameters(0, weights[1], biases[1], activations[1]);

  using Catch::Matchers::RangeEquals;
  REQUIRE_THAT(weights[1], RangeEquals(weights[0]));
  REQUIRE_THAT(biases[1], RangeEquals(biases[0]));
  REQUIRE(activations[1] == activations[0]);
}


} // namespace fluid::algorithm