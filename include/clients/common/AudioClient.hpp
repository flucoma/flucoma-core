#pragma once

#include "../../data/FluidBuffers.hpp"

namespace fluid  {
namespace client {

struct Audio {};

struct AudioIn: Audio {
  constexpr AudioIn(std::size_t n): channels(n) {}
  std::size_t channels;
};


struct AudioOut: Audio {
  constexpr AudioOut(std::size_t n): channels(n) {}
  std::size_t channels;
};
//
//struct ControlOut {
//  static constexpr std::size_t outputs = N;
//};
//
//
//template <std::size_t N>
//class BufferedAudioIn: AudioIn<N> {
//  FluidSource<double> mBuffer;
//};
//
//template <std::size_t N>
//class BufferedAudioOut: AudioOut<N> {
//  FluidSink<double> mBuffer;
//};


}
}
