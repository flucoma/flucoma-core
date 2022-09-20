#pragma once

#include "FluidIndex.hpp"
#include <Eigen/Core>
#include <memory/allocator_storage.hpp>
#include <memory/container.hpp>
#include <memory/heap_allocator.hpp>

namespace fluid {

struct FallbackAllocator
{
  using is_stateful = std::false_type;
  
  template<typename RawAlloc>
  FallbackAllocator(RawAlloc&& r):mAlloc{std::forward<RawAlloc>(r)} {}
  
  FallbackAllocator():mAlloc{foonathan::memory::heap_allocator()} {}

  void* allocate_node(std::size_t size, std::size_t x)
  {
      return mAlloc.allocate_node(size, x);
  }
  
  void deallocate_node(void* node, std::size_t size, std::size_t x) noexcept
  {
    mAlloc.deallocate_node(node,size,x);
  }
private:
  foonathan::memory::any_allocator_reference mAlloc;
};

//using Allocator = foonathan::memory::any_allocator_reference;

using Allocator = FallbackAllocator; //foonathan::memory::any_allocator_reference;


namespace rt {

using string = foonathan::memory::string<Allocator>;

template <typename T>
using vector = foonathan::memory::vector<T, Allocator>;

template <typename T>
using deque = foonathan::memory::deque<T, Allocator>;

template <typename T>
using queue = foonathan::memory::queue<T, Allocator>;
} // namespace rt

inline Allocator& FluidDefaultAllocator()
{
  static FallbackAllocator def;
  return def;
}

using ArrayXMap = Eigen::Map<Eigen::ArrayXd>;
using ArrayXXMap = Eigen::Map<Eigen::ArrayXXd>;
using ArrayXcMap = Eigen::Map<Eigen::ArrayXcd>;
using ArrayXXcMap = Eigen::Map<Eigen::ArrayXXcd>;

template <typename EigenType>
class ScopedEigenMap : public Eigen::Map<EigenType>
{
protected:
  using Eigen::Map<EigenType>::m_data;

public:
  using Scalar = typename EigenType::Scalar;
  
  template <typename Derived>
  ScopedEigenMap(const Eigen::EigenBase<Derived>& expr, Allocator& alloc):
    ScopedEigenMap(expr.rows() * expr.cols(), alloc)
  {
    *this = expr;
  }

  template <typename Derived,size_t N = Derived::NumDimensions>
  ScopedEigenMap(const Eigen::MatrixBase<Derived>& expr, Allocator& alloc,std::enable_if_t<N==1,void*> = 0)
      : ScopedEigenMap(expr.rows() * expr.cols(), alloc)
  {
    (*this).noalias() = expr;
  }
  
  template <typename Derived,size_t N = Derived::NumDimensions>
  ScopedEigenMap(const Eigen::MatrixBase<Derived>& expr, Allocator& alloc,std::enable_if_t<N==2,void*> = 0)
      : ScopedEigenMap(expr.rows(), expr.cols(), alloc)
  {
    (*this).noalias() = expr;
  }

  ScopedEigenMap(index size, Allocator& alloc)
      : Eigen::Map<EigenType>(nullptr, size),
        mStorage(asUnsigned(size), alloc)
  {
    this->m_data = mStorage.data();
  }

  ScopedEigenMap(index rows, index cols, Allocator& alloc)
      : Eigen::Map<EigenType>(nullptr, rows, cols),
        mStorage(asUnsigned(rows * cols), alloc)
  {
    this->m_data = mStorage.data();
  }

  ScopedEigenMap(ScopedEigenMap&& other) noexcept
      : Eigen::Map<EigenType>{std::move(other)},
        mStorage(std::move(other.mStorage))
  {
    this->m_data = mStorage.data();
  }

  ScopedEigenMap(const ScopedEigenMap& other)
      : Eigen::Map<EigenType>(other),
        mStorage(other.mStorage)
  {
    this->m_data = mStorage.data();
  }

  ScopedEigenMap& operator=(const ScopedEigenMap& other)
  {
    Eigen::Map<EigenType>::operator=(other);
    mStorage = other.mStorage;
    return *this;
  }

  ScopedEigenMap& operator=(ScopedEigenMap&& other) noexcept
  {
    using std::swap;
    swap(mStorage, other.mStorage);
    this->m_data = mStorage.data();
    if constexpr (Eigen::Map<EigenType>::NumDimensions == 1)
      new (static_cast<Eigen::Map<EigenType>*>(this))
          Eigen::Map<EigenType>{mStorage.data(), other.rows()};
    else
      new (static_cast<Eigen::Map<EigenType>*>(this))
          Eigen::Map<EigenType>{mStorage.data(), other.rows(), other.cols()};
    return *this;
  }


  using Eigen::Map<EigenType>::operator=;

private:
  rt::vector<Scalar> mStorage;
};

} // namespace fluid
