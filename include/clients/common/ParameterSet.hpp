#pragma once

#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"
#include "TupleUtilities.hpp"
#include <tuple>

namespace fluid {
namespace client {

/// Each parameter descriptor in the base client is a three-element tuple
/// Third element is flag indicating whether fixed (instantiation only) or not

template <typename, typename>
class ParameterDescriptorSet;

template <size_t... Os, typename... Ts>
class ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>
{
  template <bool B>
  struct FixedParam
  {
    template <typename T>
    using apply = std::is_same<Fixed<B>, typename std::tuple_element<2, T>::type>;
  };
  
  using IsFixed   = FixedParam<true>;
  using IsMutable = FixedParam<false>;
  
public:
  
  template <typename T>
  using ValueTypeAt     = typename std::tuple_element<0, T>::type::type;

  using ValueTuple      = std::tuple<ValueTypeAt<Ts>...>;
  using ValueRefTuple   = std::tuple<ValueTypeAt<Ts>&...>;
  using DescriptorType  = std::tuple<Ts...>;

  template <size_t N>
  using ParamDescriptorTypeAt = typename std::tuple_element<0,typename std::tuple_element<N, DescriptorType>::type>::type;
  
  using DescIndexList     = std::index_sequence_for<Ts...>;
  using FixedIndexList    = typename impl::FilterTupleIndices<IsFixed, DescriptorType, DescIndexList>::type;
  using MutableIndexList  = typename impl::FilterTupleIndices<IsMutable, DescriptorType, DescIndexList>::type;
  
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
    
  template <size_t N>
  auto& get() const
  {
    return std::get<0>(std::get<N>(mDescriptors));
  }
  
  template <std::size_t N>
  const char *name() const noexcept
  {
    return get<N>().name;
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

template <typename>
class ParameterSetView;

template <size_t...Os, typename... Ts>
class ParameterSetView<const ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
{
  using ParameterDescType = ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>;

protected:
    template <size_t N>
    constexpr auto descriptorAt() const
    {
        return mDescriptors.template get<N>();
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

  constexpr ParameterSetView(const ParameterDescType &d, ValueRefTuple t)
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
    using ParamType = typename ParamDescriptorTypeAt<N>::type;
      
    if (reportage) reportage->reset();
    auto &constraints   = constraintAt<N>();
    auto &param         = std::get<N>(mParams);
    const size_t offset = std::get<N>(std::make_tuple(Os...));
    ParamType x0        = std::move(x);
    ParamType x1        = constrain<offset, N>(x0, mParams, constraints, reportage);
    param = std::move(x1);
  }

  template <std::size_t N>
  auto &get() const noexcept
  {
    return std::get<N>(mParams);
  }

  template <std::size_t N>
  const char *name() const noexcept
  {
    return mDescriptors.template name<N>();
  }

  template <size_t N>
  auto defaultAt() const
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
  auto &ParamValueAt(VTuple &values)
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

  template <template <size_t, typename> class Func, size_t... Is, typename... Args>
  auto checkParameterValuesImpl(std::index_sequence<Is...> index, Args &&... args)
  {
    std::array<Result, sizeof...(Is)> results;

    ValueTuple candidateValues = std::make_tuple( Func<Is, ParamDescriptorTypeAt<Is>>()(std::forward<Args>(args)...)...);

    std::initializer_list<int>{(constrain<Os, Is>(ParamValueAt<Is>(candidateValues), candidateValues, constraintAt<Is>(), &std::get<Is>(results)), 0)...};
    
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
  
  template <size_t Offset, size_t N, typename Params, typename... Constraints>
  typename BufferT::type constrain(typename BufferT::type &thisParam, Params &, const std::tuple<Constraints...>,  Result *r)
  {
    return std::move(thisParam);
  }
  
  template <size_t Offset, size_t N, typename T, typename Params, typename... Constraints>
  T constrain(T thisParam, Params &allParams, const std::tuple<Constraints...> &c, Result *r)
  {
    // for each constraint, pass this param,all params
    return constrainImpl<Offset, N>(thisParam, allParams, c, std::index_sequence_for<Constraints...>(), r);
  }
  
  template <size_t Offset, size_t N, typename T, typename Params, typename Constraints, size_t... Is>
  T constrainImpl(T &thisParam, Params &allParams, Constraints &c, std::index_sequence<Is...>, Result *r)
  {
    T res = thisParam;
    (void) std::initializer_list<int>{(std::get<Is>(c).template clamp<Offset, N>(res, allParams, mDescriptors, r), 0)...};
    return res;
  }
  
  const ParameterDescType mDescriptors;
  ValueRefTuple mParams;
};

template <typename>
class ParameterSet
{};
  
template <size_t...Os, typename... Ts>
class ParameterSet<const ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
  : public ParameterSetView<const ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
{
  using ParameterDescType = ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>;
  using Parent = ParameterSetView<const ParameterDescType>;
  using DescIndexList = typename ParameterDescType::DescIndexList;
  using ValueTuple = typename ParameterDescType::ValueTuple;

public:
  
  constexpr ParameterSet(const ParameterDescType &d)
  : ParameterSetView<const ParameterDescType>(d, createRefTuple(DescIndexList())), mParams{create(d, DescIndexList())}
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
    
template <typename... Ts>
using ParamDescTypeFor = ParameterDescriptorSet<impl::zeroSequenceFor<Ts...>, std::tuple<Ts...>>;
    
template <typename... Args>
constexpr ParamDescTypeFor<Args...> defineParameters(Args &&... args)
{
  return {std::forward<Args>(args)...};
}

} // namespace client
} // namespace fluid
