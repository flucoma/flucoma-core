#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <flucoma/algorithms/public/NMF.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <algorithm>
#include <iostream>
#include <vector>

namespace fluid {

TEST_CASE("NMF is repeatable with user-supplied random seed")
{

  using algorithm::NMF;
  using Tensor = FluidTensor<double, 2>;
  NMF algo;

  Tensor input{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  std::vector Vs(4, Tensor(3, 3));
  std::vector Ws(4, Tensor(2, 3));
  std::vector Hs(4, Tensor(3, 2));

  algo.process(input, Ws[0], Hs[0], Vs[0], 2, 1, true, true, 42);
  algo.process(input, Ws[1], Hs[1], Vs[1], 2, 1, true, true, 42);
  algo.process(input, Ws[2], Hs[2], Vs[2], 2, 1, true, true, 5063);
  algo.process(input, Ws[3], Hs[3], Vs[3], 2, 1, true, true, 5063);

  using Catch::Matchers::RangeEquals;

  SECTION("Calls with the same seed have the same output")
  {
    REQUIRE_THAT(Ws[1], RangeEquals(Ws[0]));
    REQUIRE_THAT(Hs[1], RangeEquals(Hs[0]));
    REQUIRE_THAT(Vs[1], RangeEquals(Vs[0]));
    REQUIRE_THAT(Ws[3], RangeEquals(Ws[2]));
    REQUIRE_THAT(Hs[3], RangeEquals(Hs[2]));
    REQUIRE_THAT(Vs[3], RangeEquals(Vs[2]));
  }
  SECTION("Calls with different seeds have different outputs")
  {
    REQUIRE_THAT(Ws[1], !RangeEquals(Ws[2]));
    REQUIRE_THAT(Hs[1], !RangeEquals(Hs[2]));
    REQUIRE_THAT(Vs[1], !RangeEquals(Vs[2]));
  }
}

TEST_CASE("NMF processFrame() is repeatable with user-supplied random seed")
{
  using fluid::algorithm::NMF;
  using Tensor = fluid::FluidTensor<double, 2>;
  using Vector = fluid::FluidTensor<double, 1>;
  NMF algo;

  Vector input{{1, 0, 1, 0}};
  Tensor bases{{0, 0, 1, 0}, {1, 0, 0, 0}};
  Vector v(4);

  std::vector outputs(3, Vector(2));

  index nIter{0};
  algo.processFrame(input, bases, outputs[0], nIter, v, 42,
                    FluidDefaultAllocator());
  algo.processFrame(input, bases, outputs[1], nIter, v, 42,
                    FluidDefaultAllocator());
  algo.processFrame(input, bases, outputs[2], nIter, v, 7863,
                    FluidDefaultAllocator());

  using Catch::Matchers::RangeEquals;

  REQUIRE_THAT(outputs[1], RangeEquals(outputs[0]));
  REQUIRE_THAT(outputs[1], !RangeEquals(outputs[2]));
}
} // namespace fluid