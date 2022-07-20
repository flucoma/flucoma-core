#pragma once

#include <Eigen/Core> 
#include <memory/allocator_storage.hpp>
#include <memory/container.hpp>
#include <memory/heap_allocator.hpp>

namespace fluid {
  using Allocator  = foonathan::memory::any_allocator_reference;

  namespace rt {
    template<typename T>
    using vector = foonathan::memory::vector<T,Allocator>;

    template<typename T>
    using deque = foonathan::memory::deque<T,Allocator>;
  }
  
  Allocator& FluidDefaultAllocator()
  {
      static Allocator def = foonathan::memory::make_allocator_reference(foonathan::memory::heap_allocator());
      return def;
  }
  
  using ArrayXMap = Eigen::Map<Eigen::ArrayXd>;
  using ArrayXXMap = Eigen::Map<Eigen::ArrayXXd>;
  using ArrayXcMap = Eigen::Map<Eigen::ArrayXcd>;
  using ArrayXXcMap = Eigen::Map<Eigen::ArrayXXcd>;
  
}
