#pragma once

#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"
#include "TupleUtilities.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace impl {
/// ParamValueTypes
/// Converts a structure of parmeter declarations and constraints to a
/// structure of parameter values and constraints:
/// tuple<<pair<TypeTag,tuple<Constraints>>>> ->

///tuple<
///    tuple<Type, tuple<constraints>, FixedFlag>

/// tuple<<pair<Value,tuple<Constraints>>>>


template <size_t,typename...Ts> class ParameterDescriptorSet;

template <typename... Ts> struct ParamValueTypes {

  template <typename T>
  using ValueType = ParameterValue<typename std::tuple_element<0,T>::type>;

  template <typename T> using ConstraintsType = typename std::tuple_element<1,T>::type;

  
  using ValuePlusConstraintsType =
      std::tuple<std::pair<ValueType<Ts>, ConstraintsType<Ts>>...>;
  
  
  template<size_t O>
  constexpr static auto create(const ParameterDescriptorSet<O,Ts...> descriptors)
  {
    return ParamValueTypes::createImpl(descriptors,
                                       std::index_sequence_for<Ts...>());
  }

private:

  template<size_t N,size_t O>
  constexpr static auto descriptorAt(const ParameterDescriptorSet<O,Ts...> &d)
  {
    return std::get<0>(std::get<N>(d.descriptors()));
  }
  
  template<size_t N,size_t O>
  constexpr static auto constraintsAt(const ParameterDescriptorSet<O,Ts...> &d)
  {
    return std::get<1>(std::get<N>(d.descriptors()));
  }

  template <size_t O, size_t... Is>
  constexpr static auto createImpl(const ParameterDescriptorSet<O,Ts...> &d,
                                             std::index_sequence<Is...>)
  {
    return std::make_tuple(std::make_pair(
      ValueType<Ts>(descriptorAt<Is>(d)), constraintsAt<Is>(d))...);
  }
};

// Clamp value given constraints

template <size_t Offset, size_t N, typename T, typename Params, typename Constraints, size_t... Is>
T clampImpl(T &thisParam, Params &allParams, Constraints &c,
            std::index_sequence<Is...>, Result *r)
{
  T res = thisParam;
  (void) std::initializer_list<int>{
      (std::get<Is>(c).template clamp<Offset, N>(res, allParams, r), 0)...};
  return res;
}

//template<size_t Offset, size_t...Is>
//std::index_sequence<(Offset + Is)...> addOffset(std::index_sequence<Is...>)
//{
//  return {};
//}

template <typename T,size_t Offset>
struct Clamper
{
  template<size_t N,typename Params, typename... Constraints>
  static T clamp(T thisParam, Params &allParams, std::tuple<Constraints...> &c,
        Result *r)
  {
  // for each constraint, pass this param,all params
    return clampImpl<Offset, N>(thisParam, allParams, c,
                   std::index_sequence_for<Constraints...>(), r);
  }
};

template<size_t Offset>
struct Clamper<typename BufferT::type,Offset>
{
template<size_t N,typename Params, typename... Constraints>
static typename BufferT::type clamp(typename BufferT::type& thisParam, Params&, std::tuple<Constraints...>, Result* r)
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

template<bool B>
struct IsFixed
{
  template<typename T>
  using apply = std::is_same<Fixed<B>, typename std::tuple_element<2, T>::type>;
};

using IsFixedParamTest = IsFixed<true>;
using IsMutableParamTest = IsFixed<false>;




///Each parameter descriptor in the base client is a three-element tuple
///Third element is flag indicating whether fixed (instantiation only) or not



template<size_t CO = 0,typename...Ts> class ParameterDescriptorSet
{
  template <size_t,typename...Us> friend class ParameterDescriptorSet;
public:

  static constexpr size_t ConstraintOffset = CO;

  constexpr auto createValues() const
  {
    return impl::ParamValueTypes<Ts...>::create(*this);
  }

  using type = ParameterDescriptorSet<CO,Ts...>;

  static constexpr size_t size = sizeof...(Ts); 

  constexpr ParameterDescriptorSet(const Ts&&...ts):mDescriptors{std::make_tuple(ts...)}
  {}
  
  constexpr const std::tuple<Ts...>&  descriptors() const{ return mDescriptors; }
  
  template<size_t Offset=0, typename...Xs,size_t XOffset>
  constexpr ParameterDescriptorSet<CO + Offset + XOffset, Ts...,Xs...>
   join(const ParameterDescriptorSet<XOffset, Xs...> x)
  {
     return {mDescriptors,x.descriptors()};
  }
  
  
private:
  template<typename...Xs,typename...Ys>
  constexpr ParameterDescriptorSet(const std::tuple<Xs...>& x,const std::tuple<Ys...>& y):
  mDescriptors{std::tuple_cat(x,y)}
  {}

 
  const std::tuple<Ts...> mDescriptors;
  using DescriptorIndex = std::index_sequence_for<Ts...>;
};

} //namespace impl


template <typename> class ParameterSet;


template <template<size_t,typename...>class D, size_t O, typename...Ts>
class ParameterSet<const D<O,Ts...>>
{
  
  const impl::ParameterDescriptorSet<O,Ts...> mDescriptors;
  
public:
  
  constexpr ParameterSet(const impl::ParameterDescriptorSet<O,Ts...> &d)
      :mDescriptors{d}, mParams{impl::ParamValueTypes<Ts...>::create(d)}
  {}

  using Descriptors = const std::tuple<Ts...>;
//  using Descriptors    = const impl::ParameterDescriptorSet<Ts...> ;
  using ValueTuple     = typename impl::ParamValueTypes<Ts...>::ValuePlusConstraintsType;
//  using ParamType      = const typename std::tuple<Ts...>;
  using ParamIndexList = typename std::index_sequence_for<Ts...>;
  
  template <size_t N>
  using ParamDescriptorTypeAt = typename std::tuple_element<N, ValueTuple>::type::first_type::ParameterType;
  
  template <size_t N>
  using ParamTypeAt = typename ParamDescriptorTypeAt<N>::type;

  using FixedParams = typename impl::FilterTupleIndices<impl::IsFixedParamTest, std::decay_t<Descriptors>, ParamIndexList>::type;


  static constexpr size_t NumFixedParams = FixedParams::size();

  using MutableParams =
      typename impl::FilterTupleIndices<impl::IsMutableParamTest, std::decay_t<Descriptors>, ParamIndexList>::type;
  static constexpr size_t NumMutableParams = MutableParams::size();



  template <template <size_t N, typename T> class Func> static void iterateParameterDescriptors(Descriptors& d)
  {
    iterateParameterDescriptorsImpl<Func>(ParamIndexList());
  }

  template <template <size_t N, typename T> class Func> static void iterateFixedParameterDescriptors(Descriptors &d)
  {
    iterateParameterDescriptorsImpl<Func>(d, FixedParams());
  }

  template <template <size_t N, typename T> class Func> static void iterateMutableParameterDescriptors(Descriptors &d)
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
  
  template <template <size_t N, typename T> class Func, typename... Args> void forEachParam(Args &&... args)
  {
    forEachParamImpl<Func>(ParamIndexList(), std::forward<Args>(args)...);
  }

  template <typename T, template <size_t, typename> class Func, typename... Args> void forEachParamType(Args &&... args)
  {
    using Is = typename impl::FilterTupleIndices<IsParamType<T>, std::decay_t<Descriptors>, ParamIndexList>::type;
    forEachParamImpl<Func>(Is{}, std::forward<Args>(args)...);
  }


  void reset()
  {
//    ValueTuple freshTrousers{impl::ParamValueTypes<Ts...>::create(mDescriptors)};
//    mParams.swap(freshTrousers);
      resetImpl(ParamIndexList());
  }

  template <size_t N, typename T> void set(T &&x, Result *reportage) noexcept
  {
    if (reportage) reportage->reset();
    auto  &constraints  = std::get<N>(mParams).second;
    auto  &param       = std::get<N>(mParams).first;
    using ParamType    = typename std::remove_reference_t<decltype(param)>::type;
    auto xPrime        = impl::Clamper<ParamType,O>::template clamp<N>(x, mParams, constraints, reportage);
    param.set(std::move(xPrime));
  }

  template <std::size_t N> auto &get() noexcept { return std::get<N>(mParams).first.get(); }

  template <std::size_t N> bool changed() noexcept { return std::get<N>(mParams).first.changed(); }
  
  template <std::size_t N> const char* name() noexcept { return std::get<N>(mParams).first.name(); }
  
  template <size_t N> auto defaultAt()
  {
    return std::get<N>(mParams).first.descriptor().defaultValue;
  }

private:
  template <typename T> struct IsParamType
  {
    template <typename U> using apply = std::is_same<T, typename std::tuple_element<0, U>::type>;
  };

//  template <typename T> using ValueType = typename impl::ParamValueTypes<Ts...>::template ValueType<T>;

  //  template <size_t  Is, typename Tuple>
  //  using ParamTypeAt = typename std::tuple_element<Is, Tuple>::type;

  template <size_t N, typename VTuple> ParamTypeAt<N> &ParamValueAt(VTuple &values)
  {
    return std::get<N>(values).first.get();
  }

  template <size_t N, typename VTuple> auto &ConstraintAt(VTuple &values) { return std::get<N>(values).second; }


  template<size_t...Is>
  void resetImpl(std::index_sequence<Is...>)
  {
    std::initializer_list<int>{(std::get<Is>(mParams).first.reset(),0)...};
  }

  template <typename T, template <size_t, typename> class Func, size_t N, typename... Args>
  ParameterValue<T> makeValue(Args &&... args)
  {

    return {std::get<N>(mParams).first.descriptor(), Func<N, ParamDescriptorTypeAt<N>>()(std::forward<Args>(args)...)};
  }

  template <template <size_t, typename> class Func, size_t... Is, typename... Args>
  auto checkParameterValuesImpl(std::index_sequence<Is...> index, Args &&... args)
  {
    ValueTuple candidateValues = std::make_tuple(std::make_pair(
        makeValue<ParamDescriptorTypeAt<Is>, Func, Is>(std::forward<Args>(args)...), std::get<Is>(mParams).second)...);
    return validateParametersImpl(index, candidateValues);
  }

  template <size_t... Is> auto validateParametersImpl(std::index_sequence<Is...>, ValueTuple &values)
  {
    std::array<Result, sizeof...(Is)> results;
    std::initializer_list<int>{(impl::Clamper<ParamTypeAt<Is>,O>::template clamp<Is>(
                                    ParamValueAt<Is>(values), values, ConstraintAt<Is>(values), &std::get<Is>(results)),
                                0)...};
    return results;
  }

  template <template <size_t, typename> class Func, typename... Args, size_t... Is>
  void forEachParamImpl(std::index_sequence<Is...>, Args &&... args)
  {
    std::initializer_list<int>{(Func<Is, ParamDescriptorTypeAt<Is>>()(get<Is>(), std::forward<Args>(args)...), 0)...};
  }

  template <template <size_t N, typename T> class Op, size_t... Is>
  static void iterateParameterDescriptorsImpl(Descriptors& d,std::index_sequence<Is...>)
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

  ValueTuple mParams;
};

template<typename...Args>
constexpr impl::ParameterDescriptorSet<0,Args...> defineParameters(Args&&...args)
{
    return {std::forward<Args>(args)...};
}

//template <typename...Xs,typename...Ys>
//constexpr auto paramsCat(impl::ParameterDescriptorSet<Xs...>&& x,impl::ParameterDescriptorSet<Ys...>&& y)
//{
//  return impl::ParameterDescriptorSet<Xs...,Ys...>
//}

namespace impl
{

//  template<typename Descriptors>
//  constexpr auto parameterSetFromDescriptors(const Descriptors& d)
//  {
//    return ParameterSet<decltype(d.createValues())>{d.createValues()};
//  }
//
//  template<typename Descriptors>
//  constexpr auto parameterSetFromDescriptors(Descriptors& d)
//  {
//    return ParameterSetFromDescriptors(std::forward<Descriptors>(d),std::make_index_sequence<d.size>());
//  }



  template <typename Params, size_t O>
  class ParameterSet_Offset
  {
  public:
    ParameterSet_Offset(Params& p):mParams{p}
    {}
    
    template<size_t N>
    auto &get()
    {
      return mParams.template get<N+O>();
    }

    template <size_t N, typename T> void set(T &&x, Result *reportage) noexcept
    {
      return mParams.template set<N+O>(x,reportage);
    }
    
    template <std::size_t N> bool changed() { return mParams.template changed<N+O>(); }

  private:
    Params& mParams;
  };
}


template<size_t N, typename ParamSet>
auto& param(ParamSet& p)
{
  return p.template get<N>();
}

} // namespace client
} // namespace fluid
