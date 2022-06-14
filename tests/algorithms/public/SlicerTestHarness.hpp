#pragma once

#include <algorithms/public/STFT.hpp>
#include <data/FluidTensor.hpp>
#include <algorithm>
#include <complex>
#include <vector>

namespace fluid {

template <typename PrepareSlicerFn, typename MakeInputsFn,
          typename InvokeSlicerFn, typename Params>
std::vector<index>
SlicerTestHarness(FluidTensorView<const double, 1> testSignal, Params p,
                  PrepareSlicerFn&& prepareSlicer, MakeInputsFn&& makeInputs,
                  InvokeSlicerFn&& invokeSlicer, index addedLatency)
{
  const index            halfWindow = p.window; // >> 1;
  const index            padding = addedLatency;
  FluidTensor<double, 1> padded(p.window + halfWindow + padding +
                                testSignal.size());
  padded.fill(0);
  padded(Slice(halfWindow, testSignal.size())) <<= testSignal;
  const fluid::index nHops =
      std::floor<index>((padded.size() - p.window) / p.hop);
  auto               slicer = prepareSlicer(p);
  std::vector<index> spikePositions;
  for (index i = 0; i < nHops; ++i)
  {
    auto input = makeInputs(padded(Slice(i * p.hop, p.window)));
    if (invokeSlicer(slicer, input, p) > 0)
    {
      spikePositions.push_back((i * p.hop) - padding - p.hop);
    }
  }

  // This reproduces what the NRT wrapper does (and hence the result that the
  // existing test in SC sees). I'm dubious that
  //  it really ought to be needed though. I think we're adjusting the latency
  //  by a hop too much
  std::transform(spikePositions.begin(), spikePositions.end(),
                 spikePositions.begin(),
                 [&p](index x) { return std::max<index>(0, x); });

  spikePositions.erase(
      std::unique(spikePositions.begin(), spikePositions.end()),
      spikePositions.end());

  return spikePositions;
}


struct STFTMagnitudeInput
{

  template <typename Params>
  STFTMagnitudeInput(const Params& p)
      : mSTFT(index(p.window), index(p.fft), index(p.hop)),
        mSTFTFrame((index(p.fft) / 2) + 1), mMagnitudes((index(p.fft) / 2) + 1)
  {}

  FluidTensorView<const double, 1> operator()(FluidTensorView<double, 1> source)
  {
    mSTFT.processFrame(source, mSTFTFrame);
    mSTFT.magnitude(mSTFTFrame, mMagnitudes);
    return FluidTensorView<const double, 1>(mMagnitudes);
  }

private:
  algorithm::STFT                      mSTFT;
  FluidTensor<std::complex<double>, 1> mSTFTFrame;
  FluidTensor<double, 1>               mMagnitudes;
};


} // namespace fluid
