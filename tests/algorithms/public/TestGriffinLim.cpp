#define CATCH_CONFIG_MAIN

#include <flucoma/algorithms/public/GriffinLim.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <catch2/catch_all.hpp>
#include <complex> 
#include <vector> 

namespace fluid{
TEST_CASE("GriffinLim is repeatable with user-supplied random seed")
{

  using algorithm::GriffinLim;
  using Tensor = FluidTensor<std::complex<double>, 2>;

  index win = 64; 
  index fft = 64; 
  index hop = 64; 
  index bins = fft / 2 + 1; 

  //only actually interested in 1 frame of results, but need padding in algo
  Tensor raw_input(2,bins); 
  raw_input(0,index(bins/2)) = std::polar(1.0,0.0); 
  
  std::vector<Tensor> inouts(3, raw_input); 
  
  GriffinLim algo;

  algo.process(inouts[0], win, 1, win, fft, hop, 42);
  algo.process(inouts[1], win, 1, win, fft, hop, 42);
  algo.process(inouts[2], win, 1, win, fft, hop, 987234);

  using Catch::Matchers::RangeEquals;

  SECTION("Calls with the same seed have the same output")
  {
    REQUIRE_THAT(inouts[1], RangeEquals(inouts[0]));
  }
  SECTION("Calls with different seeds have different outputs")
  {
    REQUIRE_THAT(inouts[1], !RangeEquals(inouts[2]));
  }
}
}
