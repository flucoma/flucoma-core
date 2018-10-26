
#define EIGEN_USE_BLAS

#include <random>
#include <vector>

#include <algorithms/TransientExtraction.hpp>
#include "HISSTools_AudioFile/IAudioFile.h"
#include "HISSTools_AudioFile/OAudioFile.h"

// ******************* Parameters ******************* //

// The main blocking parameters (expose in ms for the block and pad, possiby also model order as the meaning is SR dependant)

int paramOrder = 200;       // The model order (higher == better quality + more CPU - should be quite a bit smaller than the block size)
int paramBlockSize = 2048;  // The main block size for processing (higher == longer processing times N^2 but better quality)
int paramPad = 1024;        // The analysis is done on a longer segment than the block, with this many extra values on either side
                            // This ensures the analysis is valid across the whole block (some padding is a good idea, but not too much)

// The detection parameters

// Detection is based on absolute forward and backwards prediction errors in relation to the estimated deviation of the AR model - these predictions are smoothed with a window and subjected to an on and off threshold - higher on thresholds make detection less likely and the reset threshold is used (along with a hold time) to ensure that the detection does not switch off before the end of a transient

double paramDetectPower = 1.0;           // The power factor used when windowing - higher makes detection more likely
double paramDetectThreshHi = 3.0;        // The threshold for detection (in multiples of the model deviation)
double paramDetectThreshLo = 1.1;        // The reset threshold to end a detected segment (in multiples of the model deviation)
double paramDetectHalfWindow = 7;        // Half the window size used to smooth detection functions (in samples)
int paramDetectHold = 25;               // The hold time for detection (in samples)

// This is broken right now - SET FALSE AND DO NOT EXPOSE - turning it on will produce worse results (should make things better)

const bool paramRefine = false;

// These parameters relate to the way we estimate the AR parameters robustly from data
// They probably aren't worth exposing to the end user (DO NOT EXPOSE THE ROBUST FACTOR)

int paramIterations = 3;           // How many times to iterate over the data to robustify it (can be 0)
double paramRobustFactor = 3.0;    // Data futher than this * deviation from the expected value it will be clipped

// This is just for the test app, but is used to turn the corruption process on/off if you want to test purely on the inherent transients or fake transients

bool paramCorruptInput = false;

// ************************************************** //

// Corrupts the input artificially

void corruptInput(double *output, const double *input, int size)
{
  // Simple markov chain for probability of switching states

  double pOn = paramCorruptInput ? 0.00006 : 0.0;
  double pStay = 0.989;
  double dev = 0.1;

  bool corrupt = false;
  unsigned int seed = std::random_device()();
  std::mt19937_64 randomGenerator(seed);
  std::normal_distribution<double> gaussian(0.0, dev);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  for (int i = 0; i < size; i++)
  {
    corrupt = uniform(randomGenerator) < (corrupt ? pStay : pOn);
    output[i] = corrupt ? input[i] + gaussian(randomGenerator) : input[i];
  }
}

// Write an audiofile

void writeFile(const char *name, const double *output, int size, double samplingRate)
{
  using namespace HISSTools;

  OAudioFile file(name, BaseAudioFile::kAudioFileWAVE, BaseAudioFile::kAudioFileFloat32, 1, samplingRate);

  file.writeChannel(output, size, 0);
}

// Main

int main(int argc, const char * argv[])
{
  using fluid::algorithm::TransientExtraction;

  if (argc < 5)
  {
    std::cout << "Not enough file paths specified\n";
    return -1;
  }

  const char *inputPath = argv[1];
  const char *corruptedOutputPath = argv[2];
  const char *fixedOutputPath = argv[3];
  const char *transientsOutputPath = argv[4];

  HISSTools::IAudioFile file(inputPath);

  if (!file.isOpen())
  {
    std::cout << "File could not be opened\n";
    return -2;
  }

  int frames = file.getFrames();
  auto samplingRate = file.getSamplingRate();

  std::vector<double> input(frames, 0.0);
  std::vector<double> corrupted(frames + paramBlockSize, 0.0);
  std::vector<double> fixed(frames + paramBlockSize, 0.0);
  std::vector<double> transients(frames + paramBlockSize, 0.0);

  file.readChannel(input.data(), frames, 0);
  corruptInput(corrupted.data(), input.data(), frames);

  TransientExtraction extractor(paramOrder, paramIterations, paramRobustFactor, paramRefine);

  extractor.prepareStream(paramBlockSize, paramPad);
  extractor.setDetectionParameters(paramDetectPower, paramDetectThreshHi, paramDetectThreshLo, paramDetectHalfWindow, paramDetectHold);

  std::cout << "Starting Extraction\n";

  for (int i = 0, hopSize = extractor.hopSize(); i < frames; i += hopSize)
  {
    int size = std::min(extractor.inputSize(), frames - i);
    extractor.extract(transients.data() + i, fixed.data() + i, corrupted.data() + i, size);
    std::cout << "Done " << (100 * (i + hopSize) / frames) << "%\n";
  }

  std::cout << "Finished Extraction\n";

  // Write files

  writeFile(corruptedOutputPath, corrupted.data(), frames, samplingRate);
  writeFile(fixedOutputPath, fixed.data(), frames, samplingRate);
  writeFile(transientsOutputPath, transients.data(), frames, samplingRate);

  return 0;
}
