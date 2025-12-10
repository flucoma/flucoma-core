#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <flucoma/algorithms/public/KMeans.hpp>

namespace fluid::algorithm {

TEST_CASE("KMeans against hand worked example")
{
  FluidTensor<double, 2>      points{{0, 0}, {0.5, 0.}, {0.5, 1}, {1, 1}};
  FluidTensor<std::string, 1> ids{"0", "1", "2", "3"};

  FluidDataSet<std::string, double, 1> ds(ids, points);

  fluid::algorithm::KMeans algo;
  FluidTensor<double, 2> initialMeans{{0,0},{1,1}};   
  algo.setMeans(initialMeans); 
  algo.train(ds, 2, 2, KMeans::InitMethod::randomPartion, -1);

  FluidTensor<double, 2> means(2, 2);
  algo.getMeans(means);
  FluidTensor<index, 1> assignments(4);
  algo.getAssignments(assignments);

  auto comp = [](double x, double y) -> bool {
    return Catch::Matchers::WithinRel(x).match(y);
  };

  FluidTensor<double, 2> expected_means{{0.25, 0}, {0.75, 1.0}};

  REQUIRE_THAT(means, Catch::Matchers::RangeEquals(expected_means, comp));
  REQUIRE_THAT(assignments, Catch::Matchers::RangeEquals({0, 0, 1, 1}));
}

TEST_CASE("KMeans is reproducable with manual random seed")
{

  using Tensor = FluidTensor<double, 2>;

  Tensor      points{{0, 0}, {0.5, 0.}, {0.5, 1}, {1, 1}};
  FluidTensor<std::string, 1> ids{"0", "1", "2", "3"};
  FluidDataSet<std::string, double, 1> ds(ids, points);

  std::vector means(3,Tensor(2,2));

  auto initmethod = GENERATE(algorithm::KMeans::InitMethod::randomPartion,
                             algorithm::KMeans::InitMethod::randomPoint,
                             algorithm::KMeans::InitMethod::randomSampling);
  INFO("Init method " << static_cast<long>(initmethod)); 
  algorithm::KMeans algo; 
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
  using Catch::Matchers::WithinRel;
  auto comp = [](double x, double y) -> bool {
    return Catch::Matchers::WithinRel(x).match(y);
  };

  REQUIRE_THAT(means[1], RangeEquals(means[0], comp)); 
  REQUIRE_THAT(means[1], !RangeEquals(means[2],comp)); 
}

} // namespace fluid::algorithm