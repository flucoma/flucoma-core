#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <flucoma/algorithms/public/NMFCross.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <algorithm>
#include <iostream>
#include <vector>

TEST_CASE("NMFCross is repeatable with user-supplied random seed")
{

  using fluid::algorithm::NMFCross;
  using Tensor = fluid::FluidTensor<double, 2>;
  NMFCross algo(3);

  Tensor targetMag{{0.5, 0.4}, {0.1, 1.1}, {0.7, 0.8},
                   {0.3, 0.0}, {1.0, 0.9}, {0.2, 0.6}};
  Tensor sourceMag{{0.0, 0.4}, {0.6, 0.7}, {0.8, 0.1},
                   {1.0, 0.5}, {1.1, 0.2}, {0.9, 0.3}};

  std::vector Hs(3, Tensor(6, 6));

  algo.process(targetMag, Hs[0], sourceMag, 3, 2, 7, 42);
  algo.process(targetMag, Hs[1], sourceMag, 3, 2, 7, 42);
  algo.process(targetMag, Hs[2], sourceMag, 3, 2, 7, 5063);

  using Catch::Matchers::RangeEquals;

  SECTION("Calls with the same seed have the same output")
  {
    REQUIRE_THAT(Hs[1], RangeEquals(Hs[0]));
  }
  SECTION("Calls with different seeds have different outputs")
  {
    REQUIRE_THAT(Hs[1], !RangeEquals(Hs[2]));
  }
}
