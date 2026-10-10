#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <flucoma/algorithms/public/UMAP.hpp>
#include <flucoma/data/FluidDataSet.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <vector>

namespace fluid {
TEST_CASE("UMAP is repeatable with manually set seed")
{

  using DataSet =
      FluidDataSet<std::string, double, 1>; // boohoo for std::string nonsense
  using Tensor = FluidTensor<double, 2>;
  using IdList = FluidTensor<std::string, 1>;


  algorithm::UMAP algo;

  IdList ids(10); // don't need to care what's in here
  Tensor d{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4},
           {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}};

  // needs to be < num records, or segfault (maybe this should assert?)
  index  k = 3;
  index  dim = 2;
  index  iters = 1;
  double minDist = 0.1;
  double learnRate = 1.0;

  DataSet input(ids, d);

  std::vector embeddings(3, Tensor(d.rows(), dim));

  algo.train(input, k, dim, minDist, iters, learnRate, 42);
  algo.getEmbedding(embeddings[0]);

  algo.clear();

  algo.train(input, k, dim, minDist, iters, learnRate, 42);
  algo.getEmbedding(embeddings[1]);

  algo.clear();
  algo.train(input, k, dim, minDist, iters, learnRate, 32498);

  algo.getEmbedding(embeddings[2]);

  using Catch::Matchers::RangeEquals;
  REQUIRE_THAT(embeddings[1], RangeEquals(embeddings[0]));
  REQUIRE_THAT(embeddings[1], !RangeEquals(embeddings[2]));
}
} // namespace fluid