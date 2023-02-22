/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

/*
This program demonstrates the use of the fluid decomposition toolbox
to compute a summary of spectral features of an audio file
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
#include <cstdio>
#include <iomanip>
#include <iostream>

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

void printRow(std::string name, fluid::RealVectorView vals)
{
  using namespace std;
  cout << setw(10) << name;
  for (auto v : vals) { cout << setw(10) << setprecision(5) << v; }
  cout << endl;
}

int main(int argc, char* argv[])
{
  using namespace fluid;
  using namespace fluid::algorithm;
  using fluid::index;
  using std::cout;
  using std::endl;
  using std::setw;

  if (argc <= 1)
  {
    cout << "usage: describe input_file.wav\n";
    return 1;
  }
  const char* inputFile = argv[1];

  HISSTools::IAudioFile file(inputFile);

  index nSamples = file.getFrames();
  auto  samplingRate = file.getSamplingRate();

  if (!file.isOpen())
  {
    cout << "Input file could not be opened\n";
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
  YINFFT        yin;
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
  std::string colNames[] = {"Descriptor", "mean", "stdev", "skew",
                            "kurt",       "low",  "mid",   "high"};
  std::string shapeDescs[] = {"Centroid", "Spread",   "Skewness", "Kurtosis",
                              "Rolloff",  "Flatness", "Crest"};
  for (auto n : colNames) cout << setw(10) << n;
  cout << endl;
  printRow("Pitch", pitchStats(Slice(0, 7)));
  printRow("Pitch Conf.", pitchStats(Slice(7, 7)));
  printRow("Loudness", loudnessStats(Slice(0, 7)));
  printRow("True Peak", loudnessStats(Slice(7, 7)));
  for (int i = 0; i < 7; i++)
    printRow(shapeDescs[i], shapeStats(Slice(i * 7, 7)));
  for (int i = 0; i < 13; i++)
    printRow("MFCC" + std::to_string(i), mfccStats(Slice(i * 7, 7)));
  return 0;
}
