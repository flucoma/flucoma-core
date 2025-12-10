#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <flucoma/algorithms/public/SKMeans.hpp>

namespace fluid::algorithm {

TEST_CASE("SKMeans is reproducable with manual random seed")
{

  using Tensor = FluidTensor<double, 2>;

  Tensor      points{{0.1, 0.1}, {0.5, 0.1}, {0.5, 1}, {1, 1}};
  FluidTensor<std::string, 1> ids{"0", "1", "2", "3"};
  FluidDataSet<std::string, double, 1> ds(ids, points);

  std::vector means(3,Tensor(2,2));

  auto initmethod = GENERATE(algorithm::KMeans::InitMethod::randomPartion,
                             algorithm::KMeans::InitMethod::randomPoint,
                             algorithm::KMeans::InitMethod::randomSampling);

  algorithm::SKMeans algo; 
  algo.train(ds, 2, 1, initmethod,42);
  algo.getMeans(means[0]); 
  algo.clear(); 

  algo.train(ds, 2, 1, initmethod,42);
  algo.getMeans(means[1]); 
  algo.clear(); 

  algo.train(ds, 2, 1, initmethod,4398);
  algo.getMeans(means[2]); 
  algo.clear(); 
  
  using Catch::Matchers::RangeEquals; 

  REQUIRE_THAT(means[1], RangeEquals(means[0])); 
  REQUIRE_THAT(means[1], !RangeEquals(means[2])); 
}

} // namespace fluid::algorithm