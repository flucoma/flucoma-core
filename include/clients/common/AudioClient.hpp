/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include <cstddef>
#include <type_traits>

namespace fluid {
namespace client {

struct Audio
{};
struct AudioIn : Audio
{};
struct AudioOut : Audio
{};

template <typename T>
constexpr bool isAudioIn = std::is_base_of<AudioIn, T>::value;

template <typename T>
constexpr bool isAudioOut = std::is_base_of<AudioOut, T>::value;

template <typename T>
constexpr bool isAudio = isAudioIn<T> || isAudioOut<T>;

struct Control
{};
struct ControlOut : Control
{};

template <typename T>
constexpr bool isControlOut = std::is_base_of<ControlOut, T>::value;
template <typename T>
constexpr bool isControl = std::is_base_of<Control, T>::value;


} // namespace client
} // namespace fluid
