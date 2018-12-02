#pragma once

#include "../../data/FluidBuffers.hpp"

namespace fluid  {
namespace client {

struct Audio {};

struct AudioIn: Audio {
  constexpr AudioIn(const std::size_t N): channels(N) {}
  const std::size_t channels;
};


struct AudioOut: Audio {
  constexpr AudioOut(const std::size_t N): channels(N) {}
  const std::size_t channels;
};
//
//template <std::size_t N>
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
