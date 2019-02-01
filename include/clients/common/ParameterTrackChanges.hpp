#pragma once

#include <tuple>
#include <utility>

namespace fluid {
namespace client {

template <typename...Args> class ParameterTrackChanges
{
  using type = std::tuple<Args...>;
  using indices = std::index_sequence_for<Args...>;

public:
  bool changed(Args...args) {
    return changedImpl(std::forward<Args>(args)..., indices());
  }

private:
  template <size_t... Is> bool changedImpl(Args&&...args,std::index_sequence<Is...>)
  {
    bool allSame = true;
    std::initializer_list<int>{(allSame = allSame && (std::get<Is>(mValues) == args), 0)...};
    std::initializer_list<int>{(std::get<Is>(mValues) =args, 0)...};
    return !allSame;
  }
  
  type mValues;
};
} // namespace client
} // namespace fluid
