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
  algo.train(ds, 2, 2, KMeans::InitMethod::randomPartion);

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

} // namespace fluid::algorithm