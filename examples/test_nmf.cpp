#include <iostream>
#include <string>

#include "util/audiofile.hpp"
#include "data/FluidTensor.hpp"
#include "algorithms/NMF.hpp"
#include "algorithms/RatioMask.hpp"
#include "algorithms/STFT.hpp"

int main(int argc, char *argv[]) {
  const auto &epsilon = std::numeric_limits<double>::epsilon;

  using std::complex;
  using std::cout;
  using std::vector;

  using Eigen::Map;
  using Eigen::MatrixXcd;
  using Eigen::MatrixXd;

  using fluid::audiofile::AudioFileData;
  using fluid::audiofile::readFile;
  using fluid::audiofile::writeFile;

  using fluid::FluidTensor;
  using fluid::nmf::NMF;
  using fluid::nmf::NMFModel;
  using fluid::stft::ISTFT;
  using fluid::stft::Spectrogram;
  using fluid::stft::STFT;

  using ComplexMatrix = FluidTensor<complex<double>, 2>;
  using RealVector = FluidTensor<double, 1>;
  using RealMatrix = FluidTensor<double, 2>;

  using fluid::ratiomask::RatioMask;

  if (argc <= 3) {
    cout << "usage: test_nmf train.wav test.wav rank\n";
    return 1;
  }

  AudioFileData trainData = readFile(argv[1]);
  AudioFileData testData = readFile(argv[2]);
  int nBins = 1025;
  int fftSize = 2 * (nBins - 1) ;
  int hopSize = 128;
  int rank = std::stoi(argv[3]);
  int windowSize = 2048;
  STFT stft(windowSize, fftSize, hopSize);
  ISTFT istft(windowSize, fftSize, hopSize);
  NMF nmfProcessor(rank, 100);
  RealVector train(trainData.audio[0]);
  RealVector test(testData.audio[0]);
  Spectrogram trainSpec = stft.process(train);
  Spectrogram testSpec = stft.process(test);
  NMFModel decomposition1 = nmfProcessor.process(trainSpec.getMagnitude());
  RealMatrix W = decomposition1.getW();
  NMF nmfProcessor2(rank, 100, false);
  NMFModel decomposition2 = nmfProcessor2.process(testSpec.getMagnitude(), W);
  RatioMask mask = RatioMask(decomposition2.getMixEstimate(), 1);

  for (int i = 0; i < rank; i++) {
    RealMatrix estimate = decomposition2.getEstimate(i);
    Spectrogram result(mask.process(testSpec.mData, estimate));
    RealVector audio = istft.process(result);
    testData.audio[0] = vector<double>(audio.data(), audio.data() + audio.size());
    std::string fname = "source_" + std::to_string(i) + ".wav";
    writeFile(testData, fname.c_str());
  }
  RealMatrix X = testSpec.getMagnitude();
  RealVector out(rank);
  std::cout<<"rank"<<W.extent(1)<<std::endl;
  for (int j = 0; j < X.rows(); j++) {
      fluid::FluidTensor<double, 1> frame (X.row(j));
      nmfProcessor2.processFrame(frame, W, out);
      std::cout<<out<<std::endl;
  }


  return 0;
}
