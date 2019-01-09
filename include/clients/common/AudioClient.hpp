#pragma once

#include <cstddef>
#include <type_traits>

namespace fluid  {
namespace client {

struct Audio {};

struct AudioIn: Audio {
//  constexpr AudioIn(std::size_t n): channels(n) {}
//  std::size_t channels;
};


struct AudioOut: Audio {
//  constexpr AudioOut(std::size_t n): channels(n) {}
//  std::size_t channels;
};

template <typename T>
constexpr bool isAudioIn = std::is_base_of<AudioIn, T>::value;

template <typename T>
constexpr bool isAudioOut = std::is_base_of<AudioOut, T>::value;




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
