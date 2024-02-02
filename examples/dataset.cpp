/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

/*
This program demonstrates the use of the fluid decomposition toolbox
to produces a dataset with the summary of spectral features on files
*/

#include <AudioFile/IAudioFile.h>
#include <Eigen/Core>
#include <algorithms/public/DCT.hpp>
#include <algorithms/public/Loudness.hpp>
#include <algorithms/public/MelBands.hpp>
#include <algorithms/public/MultiStats.hpp>
#include <algorithms/public/STFT.hpp>
#include <algorithms/public/SpectralShape.hpp>
#include <algorithms/public/YINFFT.hpp>
#include <data/FluidIndex.hpp>
#include <data/FluidMemory.hpp>
#include <data/TensorTypes.hpp>
#include <data/FluidDataSet.hpp>
#include <data/FluidJSON.hpp>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

fluid::RealVector computeStats(fluid::RealMatrixView        matrix,
                               fluid::algorithm::MultiStats stats)
{
  fluid::index      dim = matrix.cols();
  fluid::RealMatrix tmp(dim, 7);
  fluid::RealVector result(dim * 7);
  stats.process(matrix.transpose(), tmp);
  for (int j = 0; j < dim; j++)
  {
    result(fluid::Slice(j * 7, 7)) <<= tmp.row(j);
  }
  return result;
}

int main(int argc, char* argv[])
{
  using namespace fluid;
  using namespace fluid::algorithm;
  using fluid::index;

  if (argc <= 2)
  {
    std::cerr << "usage: describe output_file.json input_file_1.wav input_file_2.wav...\n";
    return 1;
  }

  FluidDataSet<std::string, double, 1> dataset(168);

  const char* outputFile = argv[1];
  for (int i = 2; i < argc; i++) {
    const char* inputFile = argv[i];
    HISSTools::IAudioFile file(inputFile);

    index nSamples = file.getFrames();
    auto  samplingRate = file.getSamplingRate();

    if (!file.isOpen())
    {
      std::cerr << "input file " << inputFile << " could not be opened\n";
      return -2;
    }

    index nBins = 513;
    index fftSize = 2 * (nBins - 1);
    index hopSize = 1024;
    index windowSize = 1024;
    index halfWindow = windowSize / 2;
    index nBands = 40;
    index nCoefs = 13;
    index minFreq = 20;
    index maxFreq = 5000;

    STFT          stft{windowSize, fftSize, hopSize};
    MelBands      bands{nBands, fftSize};
    DCT           dct{nBands, nCoefs};
    YINFFT        yin{nBins, FluidDefaultAllocator()};
    SpectralShape shape(FluidDefaultAllocator());
    Loudness      loudness{windowSize};
    MultiStats    stats;

    bands.init(minFreq, maxFreq, nBands, nBins, samplingRate, windowSize);
    dct.init(nBands, nCoefs);
    stats.init(0, 0, 50, 100);
    loudness.init(windowSize, samplingRate);

    RealVector in(nSamples);
    file.readChannel(in.data(), nSamples, 0);
    RealVector padded(in.size() + windowSize + hopSize);
    index      nFrames = floor((padded.size() - windowSize) / hopSize);
    RealMatrix pitchMat(nFrames, 2);
    RealMatrix loudnessMat(nFrames, 2);
    RealMatrix mfccMat(nFrames, nCoefs);
    RealMatrix shapeMat(nFrames, 7);
    std::fill(padded.begin(), padded.end(), 0);
    padded(Slice(halfWindow, in.size())) <<= in;

    for (int i = 0; i < nFrames; i++)
    {
      ComplexVector  frame(nBins);
      RealVector     magnitude(nBins);
      RealVector     mels(nBands);
      RealVector     mfccs(nCoefs);
      RealVector     pitch(2);
      RealVector     shapeDesc(7);
      RealVector     loudnessDesc(2);
      RealVectorView window = padded(fluid::Slice(i * hopSize, windowSize));
      stft.processFrame(window, frame);
      stft.magnitude(frame, magnitude);
      bands.processFrame(magnitude, mels, false, false, true,
                        FluidDefaultAllocator());
      dct.processFrame(mels, mfccs);
      mfccMat.row(i) <<= mfccs;
      yin.processFrame(magnitude, pitch, minFreq, maxFreq, samplingRate);
      pitchMat.row(i) <<= pitch;
      shape.processFrame(magnitude, shapeDesc, samplingRate, 0, -1, 0.95, false,
                        false, FluidDefaultAllocator());
      shapeMat.row(i) <<= shapeDesc;
      loudness.processFrame(window, loudnessDesc, true, true);
      loudnessMat.row(i) <<= loudnessDesc;
    }

    RealVector  pitchStats = computeStats(pitchMat, stats);
    RealVector  loudnessStats = computeStats(loudnessMat, stats);
    RealVector  shapeStats = computeStats(shapeMat, stats);
    RealVector  mfccStats = computeStats(mfccMat, stats);

    RealVector allStats(168);

    allStats(fluid::Slice(0, 14)) <<= pitchStats;
    allStats(fluid::Slice(14, 14)) <<= loudnessStats;
    allStats(fluid::Slice(28, 49)) <<= shapeStats;
    allStats(fluid::Slice(77, 91)) <<= mfccStats;

    dataset.add(inputFile, allStats);
  }

  auto outputJSON = JSONFile(outputFile, "w");
  outputJSON.write(dataset);
  
  if (!outputJSON.ok())
  {
    std::cerr << "failed to write output to " << outputFile << "\n";
  }
  
  return 0;
}
