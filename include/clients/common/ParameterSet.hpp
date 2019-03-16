#pragma once

#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"
#include "TupleUtilities.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace impl {

// Clamp value given constraints

template <typename T>
struct Clamper
{
  template <size_t Offset, size_t N, typename Params, typename... Constraints>
  static T clamp(T thisParam, Params &allParams, const std::tuple<Constraints...> &c, Result *r)
  {
    // for each constraint, pass this param,all params
    return clampImpl<Offset, N>(thisParam, allParams, c, std::index_sequence_for<Constraints...>(), r);
  }
  
private:
  template <size_t Offset, size_t N, typename Params, typename Constraints, size_t... Is>
  static T clampImpl(T &thisParam, Params &allParams, Constraints &c, std::index_sequence<Is...>, Result *r)
  {
    T res = thisParam;
    (void) std::initializer_list<int>{(std::get<Is>(c).template clamp<Offset, N>(res, allParams, r), 0)...};
    return res;
  }
};

template<>
struct Clamper<typename BufferT::type>
{
  template <size_t Offset, size_t N, typename Params, typename... Constraints>
  static typename BufferT::type clamp(typename BufferT::type &thisParam, Params &, const std::tuple<Constraints...>, Result *r)
  {
    return std::move(thisParam);
  }
};

///// FluidBaseClientImpl
///// Common functionality for clients
//template <typename T>
//std::ostream &operator<<(std::ostream &o, T &t)
//{
//  return o << t;
//}

/// Each parameter descriptor in the base client is a three-element tuple
/// Third element is flag indicating whether fixed (instantiation only) or not

template <typename, typename>
class ParameterDescriptorSet;

template <size_t... Os, typename... Ts>
class ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>
{
  template <bool B>
  struct IsFixed
  {
    template <typename T>
    using apply = std::is_same<Fixed<B>, typename std::tuple_element<2, T>::type>;
  };
  
  using IsFixedParamTest   = IsFixed<true>;
  using IsMutableParamTest = IsFixed<false>;
  
public:
  
  template <typename T>
  using ValueTypeAt           = typename std::tuple_element<0, T>::type::type;

  using ValueTuple      = std::tuple<ValueTypeAt<Ts>...>;
  using ValueRefTuple   = std::tuple<ValueTypeAt<Ts>&...>;
  using DescriptorType  = std::tuple<Ts...>;

  template <size_t N>
  using ParamDescriptorTypeAt = typename std::tuple_element<0,typename std::tuple_element<N, DescriptorType>::type>::type;
  
  using DescIndexList       = std::index_sequence_for<Ts...>;
  using FixedIndexList      = typename impl::FilterTupleIndices<IsFixedParamTest, std::decay_t<DescriptorType>, DescIndexList>::type;
  using MutableIndexList    = typename impl::FilterTupleIndices<IsMutableParamTest, std::decay_t<DescriptorType>, DescIndexList>::type;
  
  static constexpr size_t NumFixedParams    = FixedIndexList::size();
  static constexpr size_t NumMutableParams  = MutableIndexList::size();

  constexpr ParameterDescriptorSet(const Ts &&... ts) : mDescriptors{std::make_tuple(ts...)} {}
  constexpr ParameterDescriptorSet(const std::tuple<Ts...>&& t): mDescriptors{t} {}

  constexpr size_t count() const noexcept { return countImpl(DescIndexList()); }
  
  template <template <size_t N, typename T> class Func>
  void iterate() const
  {
    iterateImpl<Func>(DescIndexList());
  }
  
  template <template <size_t N, typename T> class Func>
  void iterateFixed() const
  {
    iterateImpl<Func>(FixedIndexList());
  }
  
  template <template <size_t N, typename T> class Func>
  void iterateMutable() const
  {
    iterateImpl<Func>(MutableIndexList());
  }
  
  const DescriptorType mDescriptors;
  
private:

  template <size_t... Is>
  constexpr size_t countImpl(std::index_sequence<Is...>) const noexcept
  {
    size_t count{0};
    std::initializer_list<int>{(count = count + std::get<0>(std::get<Is>(mDescriptors)).fixedSize, 0)...};
    return count;
  }
  
  template <template <size_t N, typename T> class Op, size_t... Is>
  void iterateImpl(std::index_sequence<Is...>) const
  {
    std::initializer_list<int>{(Op<Is, ParamDescriptorTypeAt<Is>>()(std::get<0>(std::get<Is>(mDescriptors))), 0)...};
  }
};
  
} // namespace impl

template <typename>
class ParameterSetImpl;

template <size_t...Os, typename... Ts>
class ParameterSetImpl<const impl::ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
{
  using ParameterDescType = impl::ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>;

protected:
    template <size_t N>
    constexpr auto descriptorAt() const
    {
        return std::get<0>(std::get<N>(mDescriptors.mDescriptors));
    }
    
public:
  
  using DescriptorType      = typename ParameterDescType::DescriptorType;
  using ValueTuple          = typename ParameterDescType::ValueTuple;
  using ValueRefTuple       = typename ParameterDescType::ValueRefTuple;
  using ParamIndexList      = typename ParameterDescType::DescIndexList;
  using FixedIndexList      = typename ParameterDescType::FixedIndexList;
  using MutableIndexList    = typename ParameterDescType::MutableIndexList;

  template <size_t N>
  using ParamDescriptorTypeAt = typename ParameterDescType::template ParamDescriptorTypeAt<N>;

  template <size_t N>
  using ParamTypeAt = typename ParamDescriptorTypeAt<N>::type;

  constexpr ParameterSetImpl(const ParameterDescType &d, ValueRefTuple t)
  : mDescriptors{d}
  , mParams{t}
  {}
  
  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> checkParameterValues(Args &&... args)
  {
    return checkParameterValuesImpl<Func>(ParamIndexList(), std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setParameterValues(bool reportage, Args &&... args)
  {
    return setParameterValuesImpl<Func>(ParamIndexList(), reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setFixedParameterValues(bool reportage, Args &&... args)
  {
    return setParameterValuesImpl<Func>(FixedIndexList(), reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setMutableParameterValues(bool reportage, Args &&... args)
  {
    return setParameterValuesImpl<Func>(MutableIndexList(), reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void forEachParam(Args &&... args)
  {
    forEachParamImpl<Func>(ParamIndexList(), std::forward<Args>(args)...);
  }

  template <typename T, template <size_t, typename> class Func, typename... Args>
  void forEachParamType(Args &&... args)
  {
    using Is = typename impl::FilterTupleIndices<IsParamType<T>, std::decay_t<DescriptorType>, ParamIndexList>::type;
    forEachParamImpl<Func>(Is{}, std::forward<Args>(args)...);
  }

  void reset() { resetImpl(ParamIndexList()); }

  template <size_t N, typename T>
  void set(T &&x, Result *reportage) noexcept
  {
    if (reportage) reportage->reset();
    auto &constraints   = constraintAt<N>();
    auto &param         = std::get<N>(mParams);
    using ParamType     = typename std::remove_reference_t<decltype(param)>::type;
    const size_t offset = std::get<N>(std::make_tuple(Os...));
    auto xPrime         = impl::Clamper<ParamType>::template clamp<offset, N>(x, mParams, constraints, reportage);
    param = std::move(xPrime);
  }

  template <std::size_t N>
  auto &get() const noexcept
  {
    return std::get<N>(mParams);
  }

  template <std::size_t N>
  const char *name() noexcept
  {
    return std::get<0>(std::get<N>(mDescriptors.mDescriptors)).name;
  }

  template <size_t N>
  auto defaultAt()
  {
    return std::get<0>(std::get<N>(mDescriptors.mDescriptors)).defaultValue;
  }
  
  template<size_t offset>
  auto subset()
  {
    return impl::RefTupleFrom<offset>(mParams);
  }
  
private:
  template <typename T>
  struct IsParamType
  {
    template <typename U>
    using apply = std::is_same<T, typename std::tuple_element<0, U>::type>;
  };

  template <size_t N, typename VTuple>
  ParamTypeAt<N> &ParamValueAt(VTuple &values)
  {
    return std::get<N>(values);
  }
  
  template <size_t N>
  constexpr auto& constraintAt() const
  {
    return std::get<1>(std::get<N>(mDescriptors.mDescriptors));
  }

  template <size_t... Is>
  void resetImpl(std::index_sequence<Is...>)
  {
    std::initializer_list<int>{(std::get<Is>(mParams) = descriptorAt<Is>().defaultValue, 0)...};
  }

  template <typename T, template <size_t, typename> class Func, size_t N, typename... Args>
  T makeValue(Args &&... args)
  {
    return {std::get<N>(mParams).descriptor(), Func<N, ParamDescriptorTypeAt<N>>()(std::forward<Args>(args)...)};
  }

  template <template <size_t, typename> class Func, size_t... Is, typename... Args>
  auto checkParameterValuesImpl(std::index_sequence<Is...> index, Args &&... args)
  {
    std::array<Result, sizeof...(Is)> results;

    ValueTuple candidateValues = std::make_tuple(makeValue<ParamDescriptorTypeAt<Is>, Func, Is>(std::forward<Args>(args)...)...);
      
    std::initializer_list<int>{(impl::Clamper<ParamTypeAt<Is>>::template clamp<Os, Is>(ParamValueAt<Is>(candidateValues), candidateValues, constraintAt<Is>(), &std::get<Is>(results)), 0)...};
    
    return results;
  }

  template <template <size_t, typename> class Func, typename... Args, size_t... Is>
  void forEachParamImpl(std::index_sequence<Is...>, Args &&... args)
  {
    std::initializer_list<int>{(Func<Is, ParamDescriptorTypeAt<Is>>()(get<Is>(), std::forward<Args>(args)...), 0)...};
  }

  template <template <size_t, typename> class Func, typename... Args, size_t... Is>
  auto setParameterValuesImpl(std::index_sequence<Is...>, bool reportage, Args &&... args)
  {
    static std::array<Result, sizeof...(Ts)> results;

    std::initializer_list<int>{
        (set<Is>(Func<Is, ParamDescriptorTypeAt<Is>>()(std::forward<Args>(args)...), reportage ? &results[Is] : nullptr),
         0)...};

    return results;
  }

  const ParameterDescType mDescriptors;
  ValueRefTuple mParams;
};

template <typename>
class ParameterSet
{};
  
template <size_t...Os, typename... Ts>
class ParameterSet<const impl::ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
  : public ParameterSetImpl<const impl::ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
{
  using ParameterDescType = impl::ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>;
  using Parent = ParameterSetImpl<const ParameterDescType>;
  using DescIndexList = typename ParameterDescType::DescIndexList;
  using ValueTuple = typename ParameterDescType::ValueTuple;
  
  template <typename T>
  using ValueType = typename ParameterDescType::template ValueTypeAt<T>;
  
public:
  
  constexpr ParameterSet(const ParameterDescType &d)
  : ParameterSetImpl<const ParameterDescType>(d, createRefTuple(DescIndexList())), mParams{create(d, DescIndexList())}
  {}
  
private:
  
  template <size_t... Is>
  constexpr auto create(const ParameterDescType &d, std::index_sequence<Is...>) const
  {
    return std::make_tuple(Parent::template descriptorAt<Is>().defaultValue...);
  }
  
  template <size_t... Is>
  constexpr auto createRefTuple(std::index_sequence<Is...>)
  {
    return std::tie(std::get<Is>(mParams)...);
  }
  
  ValueTuple mParams;
};
    
template <typename T>
constexpr size_t zero_all() { return 0u; }
    
template<typename... Ts>
using zero_sequence_for = std::index_sequence<zero_all<Ts>()...>;
  
template <typename... Ts>
using ParamDescTypeFor = impl::ParameterDescriptorSet<zero_sequence_for<Ts...>, std::tuple<Ts...>>;
    
template <typename... Args>
constexpr ParamDescTypeFor<Args...> defineParameters(Args &&... args)
{
  return {std::forward<Args>(args)...};
}

} // namespace client
} // namespace fluid
