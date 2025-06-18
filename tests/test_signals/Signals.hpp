#pragma once

#include <flucoma/data/FluidTensor.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

namespace fluid {
namespace testsignals {

FluidTensor<double, 1>& oneImpulse();
FluidTensor<double, 2>& stereoImpulses();

FluidTensor<double, 1>& sharpSines();
FluidTensor<double, 1>& smoothSine();
FluidTensor<double, 2>& guitarStrums();
FluidTensor<double, 2>& eurorackSynth();
FluidTensor<double, 1>& monoEurorackSynth();
FluidTensor<double, 2>& drums();
FluidTensor<double, 1>& monoDrums();
FluidTensor<double, 1>& monoImpulses(); 

std::vector<index> stereoImpulsePositions(); 
} // namespace testsignals
} // namespace fluid
