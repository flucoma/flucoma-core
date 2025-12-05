#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <flucoma/algorithms/util/RTPGHI.hpp>
#include <flucoma/data/FluidMemory.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <complex>
#include <vector>

namespace fluid {


TEST_CASE("RTPGHI is repeatable with manually set random seed")
{
  using Tensor = fluid::FluidTensor<double, 1>;
  using ComplexTensor = fluid::FluidTensor<std::complex<double>, 1>;
  using fluid::algorithm::RTPGHI;

  index win = 64;
  index fft = 64;
  index hop = 64;
  index bins = fft / 2 + 1;

  double mag = 1.0;
  // to stop algo converging, bypass loop by setting massive tolerence
  double tol = 2.0 * mag;

  RTPGHI algo(fft, FluidDefaultAllocator());

  Tensor input(bins);
  input[index(bins / 2)] = mag;
  std::vector results(3, ComplexTensor(bins));

  // algo has memory, so re-init after each call to test repeatability, and call
  // twice to actually generate some action
  auto runit = [&](size_t run, index seed) {
    algo.init(fft);
    algo.processFrame(input, results[run], win, fft, hop, tol, seed,
                      FluidDefaultAllocator());
    algo.processFrame(input, results[run], win, fft, hop, 2.0, seed,
                      FluidDefaultAllocator());
  };

  for (size_t run = 0; run < results.size(); ++run)
    runit(run, run < 2 ? 42 : 8347);

  using Catch::Matchers::RangeEquals;

  REQUIRE_THAT(results[0], RangeEquals(results[1]));
  REQUIRE_THAT(results[0], !RangeEquals(results[2]));
}
} // namespace fluid