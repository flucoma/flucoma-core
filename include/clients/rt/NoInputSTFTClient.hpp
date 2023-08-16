/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <random>
namespace fluid {
namespace client {

struct NoInputSTFTParams
{
    FFTParam fftSettings{1024};
    LongParam<Fixed<true>> maxFFTSize{16384};
};

class NoInputSTFTClient 
{

public:

  NoInputSTFTClient(NoInputSTFTParams& p)
      : mParams(p), mSTFTBufferedProcess{params.maxFFTSize, 0, 1}
  {
    audioChannelsIn(0);
    audioChannelsOut(1);
  }

  index latency() const { return params.fftSettings.winSize(); }

  void reset(FluidContext&) { mSTFTBufferedProcess.reset(); }

  template <typename T>
  void process(std::vector<HostVector<T>>&,
               std::vector<HostVector<T>>& output, NoInputSTFTParams& params, FluidContext& c)
  {
    mSTFTBufferedProcess.processOutput(
        mParams, output, c, [](ComplexMatrixView out) {
          //Make some noise
          std::random_device rnd_device;
          std::mt19937 mersenne_engine {rnd_device()};  
          std::uniform_real_distribution<double> dist {-1.0, 1.0};
          auto gen = [&dist, &mersenne_engine](){return dist(mersenne_engine);};    
          std::generate(begin(out), end(out), gen);
        });
  }

private:
  STFTBufferedProcess<ParamSetViewType, kFFT, true> mSTFTBufferedProcess;
};

auto NoInputSTFTInterface = makeClient<NoInputSTFTParams, NoInputSTFTClient>(
    Control(&NoInputSTFTParams::fftSettings, "fftSettings",  "FFT Settings",-1, -1),
    Control(&NoInputSTFTParams::maxFFTSize, "maxFFTSize",  "Maximum FFT Size", Min(4), PowerOfTwo{})
);


} // namespace client
} // namespace fluid
