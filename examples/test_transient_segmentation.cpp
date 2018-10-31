
#define EIGEN_USE_BLAS

#include <random>
#include <vector>

#include <HISSTools_AudioFile/IAudioFile.h>
#include <HISSTools_AudioFile/OAudioFile.h>
#include <algorithms/public/TransientSegmentation.hpp>

// ******************* Parameters ******************* //

// The main blocking parameters (expose in ms for the block and pad, possiby
// also model order as the meaning is SR dependant)

int paramOrder = 200; // The model order (higher == better quality + more CPU -
                      // should be quite a bit smaller than the block size)
int paramBlockSize = 2048; // The main block size for processing (higher ==
                           // longer processing times N^2 but better quality)
int paramPad = 1024; // The analysis is done on a longer segment than the block,
                     // with this many extra values on either side This ensures
                     // the analysis is valid across the whole block (some
                     // padding is a good idea, but not too much)

// The detection parameters

// Detection is based on absolute forward and backwards prediction errors in
// relation to the estimated deviation of the AR model - these predictions are
// smoothed with a window and subjected to an on and off threshold - higher on
// thresholds make detection less likely and the reset threshold is used (along
// with a hold time) to ensure that the detection does not switch off before the
// end of a transient

double paramDetectPower = 1.0; // The power factor used when windowing - higher
                               // makes detection more likely
double paramDetectThreshHi =
    2.9; // The threshold for detection (in multiples of the model deviation)
double paramDetectThreshLo = 1.1; // The reset threshold to end a detected
                                  // segment (in multiples of the model
                                  // deviation)
double paramDetectHalfWindow =
    7; // Half the window size used to smooth detection functions (in samples)
int paramDetectHold = 500; // The hold time for detection (in samples) - CAN
                           // (and probably should be higher than for
                           // extraction)

// These parameters relate to the way we estimate the AR parameters robustly
// from data They probably aren't worth exposing to the end user (DO NOT EXPOSE
// THE ROBUST FACTOR)

int paramIterations =
    3; // How many times to iterate over the data to robustify it (can be 0)
double paramRobustFactor = 3.0; // Data futher than this * deviation from the
                                // expected value it will be clipped

// ************************************************** //

// Write an audiofile

void writeFile(const char *name, const double *output, int size,
               double samplingRate) {
  using namespace HISSTools;

  OAudioFile file(name, BaseAudioFile::kAudioFileWAVE,
                  BaseAudioFile::kAudioFileFloat32, 1, samplingRate);

  file.writeChannel(output, size, 0);
}

// Main

int main(int argc, const char *argv[]) {
  using fluid::segmentation::TransientSegmentation;

  if (argc < 3) {
    std::cout << "Not enough file paths specified\n";
    return -1;
  }

  const char *inputPath = argv[1];
  const char *markersOutputPath = argv[2];

  HISSTools::IAudioFile file(inputPath);

  if (!file.isOpen()) {
    std::cout << "File could not be opened\n";
    return -2;
  }

  int frames = file.getFrames();
  auto samplingRate = file.getSamplingRate();

  std::vector<double> input(frames, 0.0);
  std::vector<double> segmentMarkers(frames + paramBlockSize, 0.0);

  file.readChannel(input.data(), frames, 0);

  TransientSegmentation segmentor(paramOrder, paramIterations,
                                  paramRobustFactor);

  segmentor.prepareStream(paramBlockSize, paramPad);
  segmentor.setDetectionParameters(paramDetectPower, paramDetectThreshHi,
                                   paramDetectThreshLo, paramDetectHalfWindow,
                                   paramDetectHold);

  std::cout << "Starting Segmentation\n";

  for (int i = 0, hopSize = segmentor.hopSize(); i < frames; i += hopSize) {
    int size = std::min(segmentor.inputSize(), frames - i);
    const double *markers = segmentor.process(input.data() + i, size);
    std::copy(markers, markers + hopSize, segmentMarkers.data() + i);
    std::cout << "Done " << (100 * (i + hopSize) / frames) << "%\n";
  }

  std::cout << "Finished Segmentation\n";

  // Write file

  writeFile(markersOutputPath, segmentMarkers.data(), frames, samplingRate);

  return 0;
}
