/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"
#include "TupleUtilities.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include <functional>
#include <tuple>
#include <type_traits>
		


namespace fluid {
namespace client {

/// Each parameter descriptor in the base client is a three-element tuple
/// Third element is flag indicating whether fixed (instantiation only) or not

template <typename, typename>
class ParameterDescriptorSet;

template <size_t... Os, typename... Ts>
class ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>
{
  
  
  template<size_t N, typename Tuple>
  using tuple_element_t = typename std::tuple_element<N,Tuple>::type;
  
  struct IsPrimary
  {
    template <typename T>
    using apply = std::is_same<Primary,tuple_element_t<2,T>>;
  };
  
  template <bool B>
  struct FixedParam
  {
    template <typename T>
    using apply = std::is_same<Fixed<B>, tuple_element_t<2,T>>;
  };
  
  struct IsFixed
  {
    template <typename T>
    using apply = std::is_same<Fixed<true>, tuple_element_t<2,T>>;
  };
  
  struct IsMutable
  {
    template <typename T>
    using apply = std::disjunction<
      std::is_same<Fixed<false>, tuple_element_t<2,T>>,
      std::is_same<Primary, tuple_element_t<2,T>>>;  
  };
  
    
//  using IsFixed = FixedParam<true>;
//  using IsMutable = FixedParam<false>;

  struct IsRelational
  {
    template <typename T>
    using apply = std::is_base_of<impl::Relational, T>;
  };

  struct IsNonRelational
  {
    template <typename T>
    using apply =
        std::integral_constant<bool,
                               !(std::is_base_of<impl::Relational, T>::value)>;
  };
  
  template <typename T>
  struct IsParamType
  {
    template <typename U>
    using apply = std::is_same<T, typename std::tuple_element<0, U>::type>;
  };


  template <typename T>
  using DefaultValue = decltype(std::declval<T>().defaultValue);

public:
  template <typename T>
  using ValueType = typename std::tuple_element<0, T>::type::type;

  using ValueTuple = std::tuple<ValueType<Ts>...>;
  using ValueRefTuple = std::tuple<std::reference_wrapper<ValueType<Ts>>...>;
  using DescriptorType = std::tuple<Ts...>;

  template <size_t N>
  using ParamType = typename std::tuple_element<
      0, typename std::tuple_element<N, DescriptorType>::type>::type;

  // clang < 3.7: index_sequence_for doesn't work here
  using IndexList = std::make_index_sequence<sizeof...(Ts)>;
  
  using PrimaryIndexList = typename impl::FilterTupleIndices<IsPrimary, DescriptorType,IndexList>::type;
  
  using FixedIndexList =
      typename impl::FilterTupleIndices<IsFixed, DescriptorType,
                                        IndexList>::type;
  using MutableIndexList =
      typename impl::FilterTupleIndices<IsMutable, DescriptorType,
                                        IndexList>::type;

  template <typename T, typename List>
  using RelationalList =
      typename impl::FilterTupleIndices<IsRelational, T, List>::type;

  template <typename T, typename List>
  using NonRelationalList =
      typename impl::FilterTupleIndices<IsNonRelational, T, List>::type;


  template<typename T>
  constexpr static index NumOfType = impl::FilterTupleIndices<IsParamType<T>, DescriptorType, IndexList>::type::size();

  template <typename T>
  constexpr index NumOf() const
  {
    return
        typename impl::FilterTupleIndices<T, DescriptorType, IndexList>::type::size();
  }

  static constexpr index NumPrimaryParams = PrimaryIndexList::size();
  static constexpr index NumFixedParams = FixedIndexList::size();
  static constexpr index NumMutableParams = MutableIndexList::size();

  constexpr ParameterDescriptorSet(const Ts&&... ts)
      : mDescriptors{std::make_tuple(ts...)}
  {}
  constexpr ParameterDescriptorSet(const std::tuple<Ts...>& t)
      : mDescriptors{t}
  {}

  constexpr index size() const noexcept { return sizeof...(Ts); }
  constexpr index count() const noexcept { return countImpl(IndexList()); }

  template <template <size_t N, typename T> class Func, typename... Args>
  void iterate(Args&&... args) const
  {
    iterateImpl<Func>(IndexList(), std::forward<Args>(args)...);
  }

  template<class Func>
  void iteratePrimary(Func&& func) const
  {
    ForThese(mDescriptors,
            [](auto& d, auto idx, auto&& f){ f(std::get<0>(d),idx); },
            PrimaryIndexList(), std::forward<Func>(func));
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
  constexpr auto& get() const
  {
    return std::get<0>(std::get<N>(mDescriptors));
  }

  constexpr const DescriptorType& descriptors() const { return mDescriptors; }


  template <size_t N>
  std::enable_if_t<std::is_same<typename ParamType<N>::type, rt::string>::value,
                   typename ParamType<N>::type>
  makeValue(Allocator& alloc)
  {
    return rt::string{alloc};
  }

  template <size_t N>
  std::enable_if_t<isDetected<DefaultValue, ParamType<N>>::value,
                   typename ParamType<N>::type>
  makeValue(Allocator&) const
  {
    return std::get<0>(std::get<N>(mDescriptors)).defaultValue;
  }

  template <size_t N>
  std::enable_if_t<!isDetected<DefaultValue, ParamType<N>>::value,
                   typename ParamType<N>::type>
  makeValue(Allocator&) const
  {
    return typename ParamType<N>::type{};
  }

private:
  const DescriptorType mDescriptors;

  template <size_t... Is>
  constexpr index countImpl(std::index_sequence<Is...>) const noexcept
  {
    index count{0};
    (void) std::initializer_list<int>{
        (count = count + std::get<0>(std::get<Is>(mDescriptors)).fixedSize,
         0)...};
    return count;
  }

  template <template <size_t N, typename T> class Op, typename... Args,
            size_t... Is>
  void iterateImpl(std::index_sequence<Is...>, Args&&... args) const
  {
    (void) std::initializer_list<int>{
        (Op<Is, ParamType<Is>>()(std::get<0>(std::get<Is>(mDescriptors)),
                                 std::forward<Args>(args)...),
         0)...};
  }
};

template <typename>
class ParameterSetView;

template <size_t... Os, typename... Ts>
class ParameterSetView<
    const ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
{
  enum ConstraintTypes { kAll, kNonRelational, kRelational };

protected:
  using DescriptorSetType =
      ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>;

  template <size_t N>
  constexpr auto descriptor() const
  {
    return mDescriptors.get().template get<N>();
  }

  using DescriptorType = typename DescriptorSetType::DescriptorType;
  
  using ValueRefTuple = typename DescriptorSetType::ValueRefTuple;
  using IndexList = typename DescriptorSetType::IndexList;
  using PrimaryIndexList = typename DescriptorSetType::PrimaryIndexList;
  using FixedIndexList = typename DescriptorSetType::FixedIndexList;
  using MutableIndexList = typename DescriptorSetType::MutableIndexList;

  template <typename T, typename List>
  using RelationalList =
      typename DescriptorSetType::template RelationalList<T, List>;

  template <typename T, typename List>
  using NonRelationalList =
      typename DescriptorSetType::template NonRelationalList<T, List>;

  template <size_t N>
  using ParamType = typename DescriptorSetType::template ParamType<N>;

public:

  using ValueTuple = typename DescriptorSetType::ValueTuple;

  constexpr ParameterSetView(const DescriptorSetType& d, ValueRefTuple t)
      : mDescriptors{d}, mKeepConstrained(false), mParams{t}
  {}

  ParameterSetView(ParameterSetView&& x)
      : mDescriptors{std::move(x.mDescriptors)},
        mKeepConstrained{x.mKeepConstrained}, mParams{std::move(x.mParams)}
  {}

  ParameterSetView& operator=(ParameterSetView&& x)
  {

    mDescriptors = x.mDescriptors;
    mKeepConstrained = x.mKeepConstrained;
    mParams = x.mParams;
    return *this;
  }

  auto keepConstrained(bool keep)
  {
    std::array<Result, sizeof...(Ts)> results;

    if (keep && !mKeepConstrained) results = constrainParameterValues();

    mKeepConstrained = keep;
    return results;
  }

  template <size_t N>
  typename ParamType<N>::type applyConstraintsTo(typename ParamType<N>::type x) const
  {
    const index offset = std::get<N>(std::make_tuple(Os...));
    auto&       constraints = constraint<N>();
    return constrain<offset, N, kAll>(x, constraints, nullptr);
  }

  template <size_t N, typename = std::enable_if_t<
    std::is_same_v<typename ParamType<N>::type, LongRuntimeMaxParam>, index>>
  auto applyConstraintToMax(index x) const
  {
    const auto constraints = GetIncreasingConstraints(constraint<N>());
    if constexpr (std::tuple_size<decltype(constraints)>::value)
    {
      static constexpr index offset = std::get<N>(std::make_tuple(Os...));
      std::apply(
          [&x, this](auto f) {
            f.template clamp<offset, N>(x, mParams, mDescriptors.get(),
                                        nullptr);
          },
          constraints);
    }
    return x;
  }

  std::array<Result, sizeof...(Ts)> constrainParameterValues()
  {
    return constrainParameterValuesImpl(IndexList(), IndexList());
  }

  void constrainParameterValuesRT(std::array<Result, sizeof...(Ts)>* results)
  {
    return constrainParameterValuesRTImpl(results, IndexList(), IndexList());
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void setParameterValuesRT(std::array<Result, sizeof...(Ts)>* reportage,
                            Args&&... args)
  {
    setParameterValuesRTImpl<Func>(IndexList(), reportage,
                                   std::forward<Args>(args)...);
  }


  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setParameterValues(bool reportage,
                                                       Args&&... args)
  {
    return setParameterValuesImpl<Func>(IndexList(), reportage,
                                        std::forward<Args>(args)...);
  }


  template<typename Func,typename... Args>
  std::array<Result, PrimaryIndexList::size()>
  setPrimaryParameterValues(bool reportage, Func&& func, Args&&... args)
  {
    static std::array<Result, PrimaryIndexList::size()> results;
    results.fill(Result(Result::Status::kOk)); 
    
    ForThese(mParams,[this,reportage,n=0u](auto&, auto idx, auto& res, Func&& f, auto&&... As) mutable
    {
        this->template set<idx()>(f(idx,std::forward<Args>(As)...),reportage ? &res[n++] : nullptr);
    }, PrimaryIndexList(), results, std::forward<Func>(func), std::forward<Args>(args)...);
    
    
    return constrainParameterValuesImpl(std::make_index_sequence<PrimaryIndexList::size()>(),
                                 PrimaryIndexList());  
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, FixedIndexList::size()>
  setFixedParameterValues(bool reportage, Args&&... args)
  {
    setParameterValuesImpl<Func>(FixedIndexList(), reportage,
                                 std::forward<Args>(args)...);
    return constrainParameterValuesImpl(
        std::make_index_sequence<FixedIndexList::size()>(), FixedIndexList());
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, MutableIndexList::size()>
  setMutableParameterValues(bool reportage, Args&&... args)
  {
    return setParameterValuesImpl<Func>(MutableIndexList(), reportage,
                                        std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void forEachParam(Args&&... args)
  {
    forEachParamImpl<Func>(IndexList(), std::forward<Args>(args)...);
  }

  template <typename T, template <size_t, typename> class Func,
            typename... Args>
  void forEachParamType(Args&&... args)
  {
    using Is = typename impl::FilterTupleIndices<
        IsParamType<T>, std::decay_t<DescriptorType>, IndexList>::type;
    forEachParamImpl<Func>(Is{}, std::forward<Args>(args)...);
  }
  
  //lambda version
  template <typename T, class Func,
            typename... Args>
  void forEachParamType(Func&& f, Args&&... args)
  {
    using Is = typename impl::FilterTupleIndices<
        IsParamType<T>, std::decay_t<DescriptorType>, IndexList>::type;
    
    ForThese(mParams,std::forward<Func>(f),Is{},std::forward<Args>(args)...);
        
    // /forEachParamImpl<Func>(Is{}, std::forward<Args>(args)...);
  }
  
  
  

  void reset() { resetImpl(IndexList()); }

  template <size_t N>
  void set(typename ParamType<N>::type&& x, Result* reportage) noexcept
  {
    if (reportage) reportage->reset();
    auto&       constraints = constraint<N>();
    auto&       param = std::get<N>(mParams);
    const index offset = std::get<N>(std::make_tuple(Os...));
    param.get() = mKeepConstrained
                      ? constrain<offset, N, kAll>(x, constraints, reportage)
                      : x;
  }

  template <std::size_t N>
  auto& get() const
  {
    return std::get<N>(mParams).get();
  }

  template <typename T>
  auto& get() const
  {
    return std::get<T>(mParams).get();
  }

  template <size_t offset>
  auto subset()
  {
    return impl::RefTupleFrom<offset>(mParams);
  }

  template <size_t N, typename F>
  void addListener(F&&, void*)
  {} // no-op for non-shared parameter set?

  template <size_t N>
  void removeListener(void*)
  {} // no-op for non-shared parameter set?

  template <size_t N>
  auto descriptorAt() const
  {
    return descriptor<N>();
  }

  template <typename... Us>
  void fromTuple(std::tuple<Us...> const& vals)
  {
    static_assert(sizeof...(Ts) >= sizeof...(Us), "Value tuple too big");
    // fill from back, because this might be a padded parameter set via NRT
    // wrapper
    ReverseForEach(vals, [this](auto& x, auto N)
    {
      std::get<decltype(N)::value>(mParams).get() = x;
    });
  }

  ValueTuple toTuple() { return toTupleImpl(IndexList()); }

  template<typename T>
  static constexpr index NumOfType()
  {
    return
        typename impl::FilterTupleIndices<IsParamType<T>, DescriptorType, IndexList>::size();

  }

private:

  template <typename T>
  struct IsParamType
  {
    template <typename U>
    using apply = std::is_same<T, typename std::tuple_element<0, U>::type>;
  };

  template <size_t N, typename VTuple>
  auto& paramValue(VTuple& values)
  {
    return std::get<N>(values).get();
  }

  template <size_t N>
  constexpr auto& constraint() const
  {
    return std::get<1>(std::get<N>(mDescriptors.get().descriptors()));
  }

  template <size_t... Is>
  void resetImpl(std::index_sequence<Is...>)
  {
    std::initializer_list<int>{
        (std::get<Is>(mParams) = descriptor<Is>().defaultValue, 0)...};
  }

  template <template <size_t, typename> class Func, typename... Args,
            size_t... Is>
  void forEachParamImpl(std::index_sequence<Is...>, Args&&... args)
  {
    (void) std::initializer_list<int>{
        (Func<Is, ParamType<Is>>()(get<Is>(), std::forward<Args>(args)...),
         0)...};
  }

  template <template <size_t, typename> class Func, typename... Args,
            size_t... Is>
  void setParameterValuesRTImpl(std::index_sequence<Is...>,
                                std::array<Result, sizeof...(Ts)>* reportage,
                                Args&&... args)
  {

    (void) std::initializer_list<int>{
        (set<Is>(Func<Is, ParamType<Is>>()(std::forward<Args>(args)...),
                 reportage ? reportage->data() + Is : nullptr),
         0)...};
  }



#ifdef _MSC_VER
#pragma warning(disable : 4100) // unused params on Args pack contents; don't
                                // know why,but it's not true
#endif
  template <template <size_t, typename> class Func, typename... Args,
            size_t... Is>
  auto setParameterValuesImpl(std::index_sequence<Is...>, bool reportage,
                              Args&&... args)
  {
    static std::array<Result, sizeof...(Ts)> results;

    static_cast<void>(reportage);

    (void) std::initializer_list<int>{
        (set<Is>(Func<Is, ParamType<Is>>()(std::forward<Args>(args)...),
                 reportage ? &results[Is] : nullptr),
         0)...};

    return results;
#ifdef _MSC_VER
#pragma warning(default : 4100)
#endif
  }

  template <size_t Offset, size_t N, ConstraintTypes C, typename T,
            typename... Constraints>
  T constrain(T& thisParam, const std::tuple<Constraints...>& c, Result* r) const
  {
    using CT = std::tuple<Constraints...>;
    // clang < 3.7: index_sequence_for doesn't work here
    using Idx = std::make_index_sequence<sizeof...(Constraints)>;
    switch (C)
    {
      // case kAll: return constrainImpl<Offset, N>(thisParam, c, Idx(), r);
    case kNonRelational:
      return constrainImpl<Offset, N>(thisParam, c,
                                      NonRelationalList<CT, Idx>(), r);
    case kRelational:
      return constrainImpl<Offset, N>(thisParam, c, RelationalList<CT, Idx>(),
                                      r);
    // kAll:
    default: return constrainImpl<Offset, N>(thisParam, c, Idx(), r);
    }
  }

  template <size_t Offset, size_t N, typename T, typename Constraints,
            size_t... Is>
  T constrainImpl(T& thisParam, Constraints& c, std::index_sequence<Is...>,
                  Result* r) const
  {
    T res = thisParam;
    static_cast<void>(r);
    (void) std::initializer_list<int>{
        (std::get<Is>(c).template clamp<Offset, N>(res, mParams,
                                                   mDescriptors.get(), r),
         0)...};
    return res;
  }

  template <size_t... Cs, size_t... Is>
  auto constrainParameterValuesImpl(std::index_sequence<Is...>,
                                    std::index_sequence<Cs...>)
  {
    std::array<Result, sizeof...(Is)> results;

    constexpr auto OffsetsTuple = std::make_tuple(Os...);


    (void) std::initializer_list<int>{
        (paramValue<Cs>(mParams) =
             constrain<std::get<Cs>(OffsetsTuple), Cs, kNonRelational>(
                 paramValue<Cs>(mParams), constraint<Cs>(),
                 &std::get<Is>(results)),
         0)...};
    (void) std::initializer_list<int>{
        (paramValue<Cs>(mParams) =
             constrain<std::get<Cs>(OffsetsTuple), Cs, kRelational>(
                 paramValue<Cs>(mParams), constraint<Cs>(),
                 &std::get<Is>(results)),
         0)...};

    return results;
  }

  template <size_t... Cs, size_t... Is>
  void
  constrainParameterValuesRTImpl(std::array<Result, sizeof...(Is)>* results,
                                 std::index_sequence<Is...>,
                                 std::index_sequence<Cs...>)
  {
    constexpr auto OffsetsTuple = std::make_tuple(Os...);

    (void) std::initializer_list<int>{
        (paramValue<Cs>(mParams) =
             constrain<std::get<Cs>(OffsetsTuple), Cs, kNonRelational>(
                 paramValue<Cs>(mParams), constraint<Cs>(),
                 results ? results->data() + Is : nullptr),
         0)...};
    (void) std::initializer_list<int>{
        (paramValue<Cs>(mParams) =
             constrain<std::get<Cs>(OffsetsTuple), Cs, kRelational>(
                 paramValue<Cs>(mParams), constraint<Cs>(),
                 results ? results->data() + Is : nullptr),
         0)...};
  }

protected:
  void refs(ValueTuple& p) { refsImpl(p, IndexList()); }

  std::reference_wrapper<const DescriptorSetType> mDescriptors;
  bool                                            mKeepConstrained;

private:
  template <size_t... Is>
  void refsImpl(ValueTuple& p, std::index_sequence<Is...>)
  {
    mParams = ValueRefTuple{std::get<Is>(p)...};
  }

  template <size_t... Is>
  ValueTuple toTupleImpl(std::index_sequence<Is...>)
  {
    return {std::allocator_arg_t{}, FluidDefaultAllocator(),
            std::get<Is>(mParams).get()...};
  }

  ValueRefTuple mParams;
};

template <typename>
class ParameterSet
{};

template <size_t... Os, typename... Ts>
class ParameterSet<
    const ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
    : public ParameterSetView<const ParameterDescriptorSet<
          std::index_sequence<Os...>, std::tuple<Ts...>>>
{
  using DescriptorSetType =
      ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>;
  using ViewType = ParameterSetView<const DescriptorSetType>;
  using IndexList = typename DescriptorSetType::IndexList;
  

public:

  using ValueTuple = typename DescriptorSetType::ValueTuple;

  ParameterSet(const DescriptorSetType& d, Allocator& alloc)
      : ViewType(d, createRefTuple(IndexList())), mParams{create(d, IndexList(),
                                                                 alloc)}
  {}

  // Copy construct / assign

  ParameterSet(const ParameterSet& p)
      : ViewType(p.mDescriptors.get(), createRefTuple(IndexList())),
        mParams{p.mParams}
  {
    this->mKeepConstrained = p.mKeepConstrained;
    ViewType::refs(mParams);
  }

  ParameterSet& operator=(const ParameterSet& p)
  {
    *(static_cast<ViewType*>(this)) =
        ViewType(p.mDescriptors.get(), createRefTuple(IndexList()));
    mParams = p.mParams;
    ViewType::refs(mParams);
    this->mKeepConstrained = p.mKeepConstrained;
    return *this;
  }

  // Move construct /assign
  ParameterSet(ParameterSet&& x) noexcept
      : ViewType{x.mDescriptors.get(), createRefTuple(IndexList())},
        mParams{std::move(x.mParams)}
  {
    ViewType::refs(mParams);
  }

  ParameterSet& operator=(ParameterSet&& x) noexcept
  {
    ViewType::operator=(std::move(x));
    mParams = std::move(x.mParams);
    ViewType::refs(mParams);
    return *this;
  }

private:
  template <size_t... Is>
  auto create(const DescriptorSetType& d, std::index_sequence<Is...>,
              Allocator&               alloc) const
  {
    return std::make_tuple(d.template makeValue<Is>(alloc)...);
  }

  template <size_t... Is>
  constexpr auto createRefTuple(std::index_sequence<Is...>)
  {
    return std::tie(std::get<Is>(mParams)...);
  }

  ValueTuple mParams;
};

template <typename... Ts>
using ParamDescTypeFor =
    ParameterDescriptorSet<impl::zeroSequenceFor<Ts...>, std::tuple<Ts...>>;

template <typename... Args>
constexpr ParamDescTypeFor<Args...> defineParameters(Args&&... args)
{
  return {std::forward<Args>(args)...};
}

template<typename...Ts>
constexpr ParamDescTypeFor<Ts...> defineParametersFromTuple(const std::tuple<Ts...>& t)
{
  return {t};
}

template <size_t N,typename P, typename... Args>
constexpr auto insertParameterAfter(ParamDescTypeFor<Args...> d,const P&& param)
{
  return defineParametersFromTuple(impl::tupleInsertAfter<N>(d.descriptors(),param));
}


auto constexpr NoParameters = defineParameters();

} // namespace client
} // namespace fluid
