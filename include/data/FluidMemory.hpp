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
    
  template<typename EigenType>
  class ScopedEigenMap: public Eigen::Map<EigenType>
  {
      using Parent = Eigen::Map<EigenType>;
    public:
      using Scalar = typename EigenType::Scalar;
            
      ScopedEigenMap(index size, Allocator& alloc)
        : Eigen::Map<EigenType>(nullptr, size),
          mStorage(asUnsigned(size), alloc)
          {
            Eigen::Map<EigenType>::m_data = mStorage.data();
          }
        
      ScopedEigenMap(index rows, index cols, Allocator& alloc)
        : Eigen::Map<EigenType>(nullptr, rows, cols), mStorage(asUnsigned(rows * cols), alloc)
        {
          Eigen::Map<EigenType>::m_data = mStorage.data();
        }
      
      ScopedEigenMap(ScopedEigenMap&& other) noexcept:
          mStorage(std::move(other.mStorage)),
           Eigen::Map<EigenType>::m_data(mStorage.data())
      {}
      
      ScopedEigenMap(const ScopedEigenMap& other):
          Eigen::Map<EigenType>(other),
          mStorage(other.mStorage),
          Eigen::Map<EigenType>::m_data(mStorage.data())
      {
      }
      
      ScopedEigenMap& operator=(const ScopedEigenMap& other){
         Eigen::Map<EigenType>::operator=(other);
         mStorage = other.mStorage;
         return *this;
      }
      
      ScopedEigenMap& operator=(ScopedEigenMap&& other) noexcept {
         Eigen::Map<EigenType>::operator=(std::move(other));
         using std::swap;
         swap(mStorage,other.mStorage);
         Eigen::Map<EigenType>::m_data(mStorage.data());
         return *this;
      }

      
      using Eigen::Map<EigenType>::operator=;
    private:
      rt::vector<Scalar> mStorage;
  };
  
}
