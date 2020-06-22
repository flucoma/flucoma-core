/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
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
  //clang 3.5 can't deal with an alias here (results in size 1 tuple below)
  template <typename T>
  struct makeInt{
    using type = int;
  }; 
  
  using type = std::tuple<Args...>;
  using signums = std::tuple<typename makeInt<Args>::type...>;
  
  //clang < 3.7 : index_sequence_for doesn't work here
  using indices = std::make_index_sequence<sizeof...(Args)>; 

public:

  constexpr ParameterTrackChanges(const Args... args) : mValues{args...} {}

  //Do we want *really* a default constructor? We're using it, but is it a good idea?
  ParameterTrackChanges() = default;
  
  constexpr ParameterTrackChanges(const ParameterTrackChanges&) noexcept = default;
  constexpr ParameterTrackChanges(ParameterTrackChanges&&) noexcept = default;
  constexpr ParameterTrackChanges& operator=(const ParameterTrackChanges&) noexcept = default;
  constexpr ParameterTrackChanges& operator=(ParameterTrackChanges&&) noexcept = default;

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
    (void) std::initializer_list<int>{
        (allSame = allSame && (get<Is>(mValues) == args), 0)...};
    (void) std::initializer_list<int>{
        (get<Is>(mSignums) =
             static_cast<int>(std::copysign(1, args - get<Is>(mValues))),
         0)...};
    (void) std::initializer_list<int>{(get<Is>(mValues) = args, 0)...};
    return !allSame;
  }

  signums mSignums;
  type    mValues;
};
} // namespace client
} // namespace fluid
