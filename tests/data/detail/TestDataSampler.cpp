#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <flucoma/data/SimpleDataSampler.hpp>
#include <flucoma/data/detail/DataSampler.hpp>
#include <vector>

namespace fluid::detail {

TEST_CASE("DataSampler gives reproduceable results with manually set seed")
{

  using Tensor = FluidTensor<index, 2>;

  index N = 64;

  std::vector train(6, Tensor(N / 2, 2));
  std::vector val(6, Tensor(N / 2, 2));

  SimpleDataSampler d(N, N, 0.5, true, 42);
  train[0] <<= *d.nextBatch();
  val[0] <<= *d.validationSet();

  using Catch::Matchers::RangeEquals;
  SECTION("reset() is repeatable with random seed")
  {
    d.reset();
    train[1] <<= *d.nextBatch();
    val[1] <<= *d.validationSet();
    REQUIRE_THAT(train[1], RangeEquals(train[0]));
    REQUIRE_THAT(val[1], RangeEquals(val[0]));
  }
  SECTION("new instance with same seed is repeaable")
  {
    d = SimpleDataSampler(N, N, 0.5, true, 42);
    train[2] <<= *d.nextBatch();
    val[2] <<= *d.validationSet();
    REQUIRE_THAT(train[2], RangeEquals(train[0]));
    REQUIRE_THAT(val[2], RangeEquals(val[0]));
  }
  SECTION("different seed gives different result")
  {
    d = SimpleDataSampler(N, N, 0.5, true, 23498);
    train[3] <<= *d.nextBatch();
    val[3] <<= *d.validationSet();
    REQUIRE_THAT(train[3], !RangeEquals(train[0]));
    REQUIRE_THAT(val[3], !RangeEquals(val[0]));
  }
  SECTION("automatic seeding gives different results")
  {
    d = SimpleDataSampler(N, N, 0.5, true, -1);
    train[4] <<= *d.nextBatch();
    val[4] <<= *d.validationSet();
    d = SimpleDataSampler(N, N, 0.5, true, -1);
    train[5] <<= *d.nextBatch();
    val[5] <<= *d.validationSet();
    REQUIRE_THAT(train[5], !RangeEquals(train[4]));
    REQUIRE_THAT(val[5], !RangeEquals(val[4]));
  }
}
} // namespace fluid::detail