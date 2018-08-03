#pragma once

#include <sndfile.h>
#include <vector>
#include <iostream>

namespace fluid {
namespace audiofile {


using std::vector;
using std::runtime_error;

struct AudioFileData {
    long numFrames;
    int numChannels;
    int sampleRate;
    std::vector<std::vector<double>> audio;
};

AudioFileData readFile(const char* path) {

  SF_INFO info;
  SNDFILE *file;
  memset(&info, 0, sizeof(info));

  file = sf_open(path, SFM_READ, &info);

  if (file == NULL) {
    throw runtime_error("Could not open input audio file\n");
  }

  if (info.channels != 1) {
    throw runtime_error("Only mono files supported for now\n");
  }

  vector<double> samples(info.frames * info.channels);
  sf_count_t num_read = sf_readf_double(file, samples.data(), info.frames);
  if (num_read != info.frames) {
    throw runtime_error("Read error\n");
  }
  sf_close(file);

  AudioFileData data = {
    info.frames,
    info.channels,
    info.samplerate,
    { samples }
  };

  return data;
}

void writeFile(AudioFileData data, const char* path) {
  SF_INFO info;
  SNDFILE *file;
  info.channels = data.numChannels;
  info.samplerate = data.sampleRate;
  info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

  file = sf_open(path, SFM_WRITE, &info);
  if (file == NULL) {
    throw runtime_error("Could not open output audio file\n");
  }

  sf_write_double(file, data.audio[0].data(), data.numFrames);
  sf_write_sync(file);
  sf_close(file);
}


}
}
