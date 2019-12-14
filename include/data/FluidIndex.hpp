#pragma once

#include <cassert> 
#include <cstddef>
#include <type_traits> 

namespace fluid{ 
  using Index = std::ptrdiff_t; 
  
  std::make_unsigned_t<Index> IndexCast(Index i)
  {
    assert(i >= 0); 
    return static_cast<std::make_unsigned_t<Index>>(i); 
  }  
}
