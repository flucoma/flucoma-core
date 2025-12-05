#define CATCH_CONFIG_MAIN
#include <flucoma/algorithms/public/NNDSVD.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <catch2/catch_all.hpp>
#include <vector> 

TEST_CASE("NNDSVD Mode 1 is repeatable with manually set random seed"){

using Tensor = fluid::FluidTensor<double,2>; 
using fluid::algorithm::NNDSVD; 

// To test the effect of randomness in NNDSVD mode 1, there must be 0s in the input
Tensor input = {{0,0,0},{0,0,0},{0,0,0}}; 

std::vector Ws(3, Tensor(3,3)); 
std::vector Hs(3, Tensor(3,3)); 

NNDSVD algo;

algo.process(input,Ws[0],Hs[0],2,2,0.8,1, 42); 
algo.process(input,Ws[1],Hs[1],2,2,0.8,1, 42); 
algo.process(input,Ws[2],Hs[2],2,2,0.8,1, 4672);

using Catch::Matchers::RangeEquals; 

REQUIRE_THAT(Ws[1],RangeEquals(Ws[0])); 
REQUIRE_THAT(Ws[1],!RangeEquals(Ws[2])); 
REQUIRE_THAT(Hs[1],RangeEquals(Hs[0])); 
REQUIRE_THAT(Hs[1],!RangeEquals(Hs[2])); 
}