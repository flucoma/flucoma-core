/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include <cmath>
#include <tuple>
#include <utility>

namespace fluid {
namespace client {

template <typename... Args>
class ParameterTrackChanges
{
  template <typename T>
  using makeInt = int;

  using type = std::tuple<Args...>;
  using signums = std::tuple<makeInt<Args>...>;
  using indices = std::index_sequence_for<Args...>;

public:
  ParameterTrackChanges() = default;
  constexpr ParameterTrackChanges(const Args... args) : mValues{args...} {}

  bool changed(Args... args)
  {
    return changedImpl(std::forward<Args>(args)..., indices());
  }

  template <size_t N>
  int direction()
  {
    return std::get<N>(mSignums);
  }

private:
  template <size_t... Is>
  bool changedImpl(Args&&... args, std::index_sequence<Is...>)
  {
    bool allSame = true;
    using std::get;
    std::initializer_list<int>{
        (allSame = allSame && (get<Is>(mValues) == args), 0)...};
    std::initializer_list<int>{(get<Is>(mSignums) = static_cast<int>(
                                    std::copysign(1, args - get<Is>(mValues))),
                                0)...};
    std::initializer_list<int>{(get<Is>(mValues) = args, 0)...};
    return !allSame;
  }

  signums mSignums;
  type    mValues;
};
} // namespace client
} // namespace fluid
