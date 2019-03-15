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
  template <size_t N, typename Params, typename... Constraints>
  static T clamp(T thisParam, Params &allParams, std::tuple<Constraints...> &c, Result *r)
  {
    // for each constraint, pass this param,all params
    return clampImpl<N>(thisParam, allParams, c, std::index_sequence_for<Constraints...>(), r);
  }
  
private:
  template <size_t N, typename Params, typename Constraints, size_t... Is>
  static T clampImpl(T &thisParam, Params &allParams, Constraints &c, std::index_sequence<Is...>, Result *r)
  {
    T res = thisParam;
    (void) std::initializer_list<int>{(std::get<Is>(c).template clamp<N>(res, allParams, r), 0)...};
    return res;
  }
};

template<>
struct Clamper<typename BufferT::type>
{
  template <size_t N, typename Params, typename... Constraints>
  static typename BufferT::type clamp(typename BufferT::type &thisParam, Params &, std::tuple<Constraints...>, Result *r)
  {
    return std::move(thisParam);
  }
};

/// FluidBaseClientImpl
/// Common functionality for clients
template <typename T>
std::ostream &operator<<(std::ostream &o, ParameterValue<T> &t)
{
  return o << t.get();
}

/// Each parameter descriptor in the base client is a three-element tuple
/// Third element is flag indicating whether fixed (instantiation only) or not

template <typename... Ts>
class ParameterDescriptorSet
{
  template <typename T>
  using ValueType = ParameterValue<typename std::tuple_element<0, T>::type>;
  
  template <typename T>
  using ConstraintsType = typename std::tuple_element<1, T>::type;
  
public:
  
  using ValuePlusConstraintsType = std::tuple<std::pair<ValueType<Ts>, ConstraintsType<Ts>>...>;

  constexpr ParameterDescriptorSet(const Ts &&... ts) : mDescriptors{std::make_tuple(ts...)}
  {}
  
  constexpr ParameterDescriptorSet(const std::tuple<Ts...>&& t): mDescriptors{t}
  {}

  constexpr size_t count() const noexcept { return countImpl(DescriptorIndex()); }

  constexpr const std::tuple<Ts...> &descriptors() const { return mDescriptors; }

  constexpr auto createValues() const { return create(DescriptorIndex()); }

private:

  template <size_t... Is>
  constexpr size_t countImpl(std::index_sequence<Is...>) const noexcept
  {
    size_t count{0};
    std::initializer_list<int>{(count = count + std::get<0>(std::get<Is>(mDescriptors)).fixedSize, 0)...};
    return count;
  }
  
  template <size_t N>
  constexpr auto descriptorAt() const
  {
    return std::get<0>(std::get<N>(descriptors()));
  }
  
  template <size_t N>
  constexpr auto constraintsAt() const
  {
    return std::get<1>(std::get<N>(descriptors()));
  }
  
  template <size_t... Is>
  constexpr auto create(std::index_sequence<Is...>) const
  {
    return std::make_tuple(std::make_pair(ValueType<Ts>(descriptorAt<Is>()), constraintsAt<Is>())...);
  }
  
  const std::tuple<Ts...> mDescriptors;
  using DescriptorIndex = std::index_sequence_for<Ts...>;
};
  
} // namespace impl

template <typename>
class ParameterSetImpl;

template <template <typename...> class D, typename... Ts>
class ParameterSetImpl<const D<Ts...>>
{
  template <bool B>
  struct IsFixed
  {
    template <typename T>
    using apply = std::is_same<Fixed<B>, typename std::tuple_element<2, T>::type>;
  };
  
  using IsFixedParamTest   = IsFixed<true>;
  using IsMutableParamTest = IsFixed<false>;
  using ParameterDescType = D<Ts...>;
  const ParameterDescType mDescriptors;

public:
  
  using Descriptors = const std::tuple<Ts...>;
  using ValueTuple = typename ParameterDescType::ValuePlusConstraintsType;
  using ParamIndexList = typename std::index_sequence_for<Ts...>;

  constexpr ParameterSetImpl(const impl::ParameterDescriptorSet<Ts...> &d, ValueTuple &t)
      : mDescriptors{d}
      , mParams{t}
  {}

  template <size_t N>
  using ParamDescriptorTypeAt = typename std::tuple_element<N, ValueTuple>::type::first_type::ParameterType;

  template <size_t N>
  using ParamTypeAt = typename ParamDescriptorTypeAt<N>::type;

  using FixedParams =
      typename impl::FilterTupleIndices<IsFixedParamTest, std::decay_t<Descriptors>, ParamIndexList>::type;

  static constexpr size_t NumFixedParams = FixedParams::size();

  using MutableParams =
      typename impl::FilterTupleIndices<IsMutableParamTest, std::decay_t<Descriptors>, ParamIndexList>::type;
  static constexpr size_t NumMutableParams = MutableParams::size();

  template <template <size_t N, typename T> class Func>
  static void iterateParameterDescriptors(Descriptors &d)
  {
    iterateParameterDescriptorsImpl<Func>(ParamIndexList());
  }

  template <template <size_t N, typename T> class Func>
  static void iterateFixedParameterDescriptors(Descriptors &d)
  {
    iterateParameterDescriptorsImpl<Func>(d, FixedParams());
  }

  template <template <size_t N, typename T> class Func>
  static void iterateMutableParameterDescriptors(Descriptors &d)
  {
    iterateParameterDescriptorsImpl<Func>(d, MutableParams());
  }

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
    return setParameterValuesImpl<Func>(FixedParams(), reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setMutableParameterValues(bool reportage, Args &&... args)
  {
    return setParameterValuesImpl<Func>(MutableParams(), reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void forEachParam(Args &&... args)
  {
    forEachParamImpl<Func>(ParamIndexList(), std::forward<Args>(args)...);
  }

  template <typename T, template <size_t, typename> class Func, typename... Args>
  void forEachParamType(Args &&... args)
  {
    using Is = typename impl::FilterTupleIndices<IsParamType<T>, std::decay_t<Descriptors>, ParamIndexList>::type;
    forEachParamImpl<Func>(Is{}, std::forward<Args>(args)...);
  }

  void reset() { resetImpl(ParamIndexList()); }

  template <size_t N, typename T>
  void set(T &&x, Result *reportage) noexcept
  {
    if (reportage) reportage->reset();
    auto &constraints = std::get<N>(mParams).second;
    auto &param       = std::get<N>(mParams).first;
    using ParamType   = typename std::remove_reference_t<decltype(param)>::type;
    auto xPrime       = impl::Clamper<ParamType>::template clamp<N>(x, mParams, constraints, reportage);
    param.set(std::move(xPrime));
  }

  template <std::size_t N>
  auto &get() noexcept
  {
    return std::get<N>(mParams).first.get();
  }

  template <std::size_t N>
  bool changed() noexcept
  {
    return std::get<N>(mParams).first.changed();
  }

  template <std::size_t N>
  const char *name() noexcept
  {
    return std::get<N>(mParams).first.name();
  }

  template <size_t N>
  auto defaultAt()
  {
    return std::get<N>(mParams).first.descriptor().defaultValue;
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
    return std::get<N>(values).first.get();
  }

  template <size_t N, typename VTuple>
  auto &ConstraintAt(VTuple &values)
  {
    return std::get<N>(values).second;
  }

  template <size_t... Is>
  void resetImpl(std::index_sequence<Is...>)
  {
    std::initializer_list<int>{(std::get<Is>(mParams).first.reset(), 0)...};
  }

  template <typename T, template <size_t, typename> class Func, size_t N, typename... Args>
  ParameterValue<T> makeValue(Args &&... args)
  {

    return {std::get<N>(mParams).first.descriptor(), Func<N, ParamDescriptorTypeAt<N>>()(std::forward<Args>(args)...)};
  }

  template <template <size_t, typename> class Func, size_t... Is, typename... Args>
  auto checkParameterValuesImpl(std::index_sequence<Is...> index, Args &&... args)
  {
    std::array<Result, sizeof...(Is)> results;

    ValueTuple candidateValues = std::make_tuple(std::make_pair(
        makeValue<ParamDescriptorTypeAt<Is>, Func, Is>(std::forward<Args>(args)...), std::get<Is>(mParams).second)...);

    std::initializer_list<int>{(impl::Clamper<ParamTypeAt<Is>>::template clamp<Is>(ParamValueAt<Is>(candidateValues), candidateValues, ConstraintAt<Is>(candidateValues), &std::get<Is>(results)), 0)...};
    
    return results;
  }

  template <template <size_t, typename> class Func, typename... Args, size_t... Is>
  void forEachParamImpl(std::index_sequence<Is...>, Args &&... args)
  {
    std::initializer_list<int>{(Func<Is, ParamDescriptorTypeAt<Is>>()(get<Is>(), std::forward<Args>(args)...), 0)...};
  }

  template <template <size_t N, typename T> class Op, size_t... Is>
  static void iterateParameterDescriptorsImpl(Descriptors &d, std::index_sequence<Is...>)
  {
    std::initializer_list<int>{(Op<Is, ParamDescriptorTypeAt<Is>>()(std::get<0>(std::get<Is>(d))), 0)...};
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

  ValueTuple& mParams;
};

template <typename T>
class ParameterSet : public ParameterSetImpl<T>
{};
  
template <template <typename...> class D, typename... Ts>
class ParameterSet<const D<Ts...>> : public ParameterSetImpl<const D<Ts...>>
{
  using ParameterDescType = D<Ts...>;
  using ValueTuple = typename ParameterDescType::ValuePlusConstraintsType;
 
public:
  
  constexpr ParameterSet(const impl::ParameterDescriptorSet<Ts...> &d)
  : ParameterSetImpl<const ParameterDescType>(d, mParams), mParams{d.createValues()}
  {}
  
private:
  
  ValueTuple mParams;
};
  
template <typename... Args>
constexpr impl::ParameterDescriptorSet<Args...> defineParameters(Args &&... args)
{
  return {std::forward<Args>(args)...};
}

template <size_t N, typename ParamSet>
auto &param(ParamSet &p)
{
  return p.template get<N>();
}

} // namespace client
} // namespace fluid
