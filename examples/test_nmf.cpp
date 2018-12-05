#include <iostream>
#include <string>

#include "util/audiofile.hpp"

#include <algorithms/public/NMF.hpp>
#include <algorithms/public/RatioMask.hpp>
#include <algorithms/public/STFT.hpp>
#include <data/FluidTensor.hpp>

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
  using fluid::algorithm::ISTFT;
  using fluid::algorithm::NMF;
  using fluid::algorithm::STFT;

  using fluid::algorithm::RatioMask;

  if (argc <= 3) {
    cout << "usage: test_nmf train.wav test.wav rank\n";
    return 1;
  }

  AudioFileData trainData = readFile(argv[1]);
  AudioFileData testData = readFile(argv[2]);
  int nBins = 1025;
  int fftSize = 2 * (nBins - 1);
  int hopSize = 128;
  int rank = std::stoi(argv[3]);
  int windowSize = 2048;
  STFT stft(windowSize, fftSize, hopSize);
  ISTFT istft(windowSize, fftSize, hopSize);
  NMF nmfProcessor(rank, 100);
  NMF nmfProcessor2(rank, 100, false);

  fluid::FluidTensor<double, 1> train(trainData.audio[0]);
  fluid::FluidTensor<double, 1> test(testData.audio[0]);
  int nFramesTrain = floor((train.size() + hopSize) / hopSize);
  int nFramesTest = floor((test.size() + hopSize) / hopSize);

  // allocation
  fluid::FluidTensor<complex<double>, 2> trainSpec(nFramesTrain, nBins);
  fluid::FluidTensor<complex<double>, 2> testSpec(nFramesTest, nBins);
  fluid::FluidTensor<double, 2> trainMag(nFramesTrain, nBins);
  fluid::FluidTensor<double, 2> testMag(nFramesTest, nBins);

  fluid::FluidTensor<double, 2> trainV(nFramesTrain, nBins);
  fluid::FluidTensor<double, 2> testV(nFramesTest, nBins);

  fluid::FluidTensor<double, 2> trainW(rank, nBins);
  fluid::FluidTensor<double, 2> testW(rank, nBins);
  fluid::FluidTensor<double, 2> trainH(nFramesTrain, rank);
  fluid::FluidTensor<double, 2> testH(nFramesTest, rank);

  // processing
  stft.process(train, trainSpec);
  stft.process(test, testSpec);
  STFT::magnitude(trainSpec, trainMag);
  STFT::magnitude(testSpec, testMag);
  nmfProcessor.process(trainMag, trainW, trainH, trainV);
  RatioMask trainMask = RatioMask(trainV, 1);

  for (int i = 0; i < rank; i++) {
    fluid::FluidTensor<double, 2> tmpV1(nFramesTrain, nBins);
    fluid::FluidTensor<complex<double>, 2> tmpSpec1(nFramesTrain, nBins);
    fluid::FluidTensor<double, 1> tmpAudio1(train.size());
    fluid::FluidTensor<double, 2> tmpMag(nFramesTrain, nBins);
    NMF::estimate(trainW, trainH, i, tmpV1);
    trainMask.process(trainSpec, tmpV1, tmpSpec1);
    STFT::magnitude(tmpSpec1, tmpMag);
    istft.process(tmpSpec1, tmpAudio1);
    trainData.audio[0] =
        vector<double>(tmpAudio1.data(), tmpAudio1.data() + tmpAudio1.size());
    std::string fname = "train_source_" + std::to_string(i) + ".wav";
    writeFile(trainData, fname.c_str());
  }

  nmfProcessor2.process(testMag, testW, testH, testV, trainW);
  RatioMask testMask = RatioMask(testV, 1);

  for (int i = 0; i < rank; i++) {
    fluid::FluidTensor<double, 2> tmpV2(nFramesTest, nBins);
    fluid::FluidTensor<complex<double>, 2> tmpSpec2(nFramesTest, nBins);
    fluid::FluidTensor<double, 1> tmpAudio2(test.size());
    NMF::estimate(testW, testH, i, tmpV2);
    testMask.process(testSpec, tmpV2, tmpSpec2);
    istft.process(tmpSpec2, tmpAudio2);
    testData.audio[0] =
        vector<double>(tmpAudio2.data(), tmpAudio2.data() + tmpAudio2.size());
    std::string fname = "test_source_" + std::to_string(i) + ".wav";
    writeFile(testData, fname.c_str());
  }


  return 0;
}
