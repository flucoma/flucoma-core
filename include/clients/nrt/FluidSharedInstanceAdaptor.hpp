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

#include "../common/FluidBaseClient.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <functional>
#include <string>
#include <unordered_map>

namespace fluid {
namespace client {

template <typename, typename>
class ParamAliasAdaptor;

template <typename NRTClient, typename... Ts>
class ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>
{
  using WrappedClient = ClientWrapper<NRTClient>;
  using ParamSetType = typename WrappedClient::ParamSetType;

  using ListenerEntry = std::pair<std::function<void()>, void*>;
  using ListenerList = std::vector<ListenerEntry>;
  using ListenersArray = std::array<ListenerList, sizeof...(Ts)>;

  struct ListeningParams : public std::enable_shared_from_this<ListeningParams>
  {
    using std::enable_shared_from_this<ListeningParams>::shared_from_this;

    ParamSetType   params{ClientWrapper<NRTClient>::getParameterDescriptors(),
                        FluidDefaultAllocator()};
    ListenersArray listeners;

    std::shared_ptr<ListeningParams> getShared() { return shared_from_this(); };
  };

  using ParamsPointer = std::shared_ptr<ListeningParams>;
  using ParamsWeakPointer = std::weak_ptr<ListeningParams>;
  using LookupTable = std::unordered_map<rt::string, ParamsWeakPointer, std::hash<std::string_view>>;

  template <size_t N>
  using ParamType =
      typename WrappedClient::ParamDescType::template ParamType<N>;

public:
  using ValueTuple = typename ParamSetType::ValueTuple;

  ParamAliasAdaptor(typename NRTClient::ParamDescType&, Allocator&)
      : mParams{std::make_shared<ListeningParams>()}
  {}

  ~ParamAliasAdaptor()
  {
    if (mParams &&
        mParams.use_count() == 1) // is this the last remaining user of this
                                  // Corpus, except the hash table?
    {
      auto name = mParams->params.template get<0>();
      mParamsTable.erase(name); // then remove it from the universe
    }
  }

  typename WrappedClient::ParamSetType& instance() { return mParams->params; }

  auto keepConstrained(bool keep)
  {
    return mParams->params.keepConstrained(keep);
  }

  std::array<Result, sizeof...(Ts)> constrainParameterValues()
  {
    return mParams->params.constrainParameterValues();
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setParameterValues(bool reportage,
                                                       Args&&... args)
  {
    return mParams->params.template setParameterValues<Func>(
        reportage, std::forward<Args>(args)...);
  }


  void constrainParameterValuesRT(std::array<Result, sizeof...(Ts)>* results)
  {
    mParams->params.constrainParameterValuesRT(results);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void setParameterValuesRT(std::array<Result, sizeof...(Ts)>* reportage,
                            Args&&... args)
  {
    mParams->params.template setParameterValuesRT<Func>(
        reportage, std::forward<Args>(args)...);
  }


  Result lookup(rt::string name)
  {
    return mParamsTable.count(name)
               ? Result{}
               : Result{Result::Status::kWarning, name, " not found"};
  }

  Result refer(rt::string name)
  {
    if (mParamsTable.count(name))
    {
      mParams = mParamsTable[name].lock()->getShared();
      return {};
    }
    else
      return {Result::Status::kWarning, name, " not found"};
  }

  template<typename Func,typename... Args>
  auto setPrimaryParameterValues(bool reportage,Func&& f, Args&&...args)
  {
     return mParams->params.setPrimaryParameterValues(reportage, std::forward<Func>(f),std::forward<Args>(args)...);
  }
  
  template <template <size_t N, typename T> class Func, typename... Args>
  auto setFixedParameterValues(bool reportage, Args&&... args)
  {
    auto results = mParams->params.template setFixedParameterValues<Func>(
        reportage, std::forward<Args>(args)...);

    rt::string name = mParams->params.template get<0>();

    if (!name.size())
    {
      results[0].addMessage("Shared object given no name – won't be shared!");
      results[0].set(Result::Status::kWarning);
      return results;
    }

    if (!mParamsTable.count(name)) // key not already in table
    {
      mParamsTable.emplace(name, mParams);
    }

    refer(name);

    return results;
  }

  template<size_t N> 
  typename ParamType<N>::type applyConstraintsTo(typename ParamType<N>::type x) const
  {
      return mParams->template applyConstraintsTo<N>(x); 
  }

  template <size_t N>
  typename ParamType<N>::type
  applyConstraintToMax(typename ParamType<N>::type x) const
  {
    return mParams->template applyConstraintToMax(x);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setMutableParameterValues(bool reportage,
                                                              Args&&... args)
  {
    return mParams->params.template setMutableParameterValues<Func>(
        reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void forEachParam(Args&&... args)
  {
    mParams->params.template forEachParam<Func>(std::forward<Args>(args)...);
  }

  template <typename T, template <size_t, typename> class Func,
            typename... Args>
  void forEachParamType(Args&&... args)
  {
    mParams->params.template forEachParamType<T, Func>(
        std::forward<Args>(args)...);
  }


  //lambda version
  template <typename T, class Func,
            typename... Args>
  void forEachParamType(Func&& f, Args&&... args)
  {
    mParams->params.template forEachParamType<T>(std::forward<Func>(f),std::forward<Args>(args)...);
  }
  
  void reset() { mParams->params.reset(); }

  template <size_t N>
  void set(typename ParamType<N>::type&& x, Result* reportage) noexcept
  {
    mParams->params.template set<N>(
        std::forward<typename ParamType<N>::type>(x), reportage);

    auto listeners = mParams->listeners[N];
    for (auto&& l : listeners) l.first();
  }

  template <std::size_t N>
  auto& get() const
  {
    return mParams->params.template get<N>();
  }

  template <size_t offset>
  auto subset()
  {
    return mParams->params.template subset<offset>();
  }
  
  template <size_t N>
  auto descriptorAt()
  {
    return mParams->params.template descriptor<N>();
  }
  
  template <typename Tuple>
  void fromTuple(Tuple const& vals)
  {
    mParams->params.fromTuple(vals);

    for (auto&& listeners : mParams->listeners)
      for (auto&& l : listeners) l.first();
  }

  typename ParamSetType::ValueTuple toTuple()
  {
    return {mParams->params.toTuple()};
  }

  template <size_t N, typename F>
  void addListener(F&& f, void* key)
  {
    mParams->listeners[N].emplace_back(ListenerEntry{std::forward<F>(f), key});
  }

  template <size_t N>
  void removeListener(void* key)
  {
    auto& listeners = mParams->listeners[N];
    listeners.erase(
        std::remove_if(listeners.begin(), listeners.end(),
                       [&key](ListenerEntry& e) { return e.second == key; }),
        listeners.end());
  }

private:
  ParamsPointer      mParams;
  static LookupTable mParamsTable;
  //    std::array<std::vector<std::function<void()>>,sizeof...(Ts)> mListeners;
};

// template<typename NRTClient, typename...Ts> //init master param set
// typename ClientWrapper<NRTClient>::ParamSetType
// ParamAliasAdaptor<NRTClient,
// std::tuple<Ts...>>::mParams{ClientWrapper<NRTClient>::getParameterDescriptors()};

// template<typename NRTClient, typename...Ts> //init parameter listeners
// std::array<std::vector<std::function<void()>>,sizeof...(Ts)>
// ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>::mListeners{};

template <typename NRTClient, typename... Ts>
typename ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>::LookupTable
    ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>::mParamsTable{};


template <typename NRTClient>
class NRTSharedInstanceAdaptor : public OfflineIn, public OfflineOut
{
  
public:
  using WrappedClient = ClientWrapper<NRTClient>;

  struct SharedClient : public NRTClient,
                        public std::enable_shared_from_this<SharedClient>
  {
    using std::enable_shared_from_this<SharedClient>::shared_from_this;
    using NRTClient::NRTClient;
    std::shared_ptr<SharedClient> shared() { return shared_from_this(); };
  };
  using ClientPointer = std::shared_ptr<SharedClient>;
  using ClientWeakPointer = typename std::weak_ptr<const SharedClient>;
  using ParamDescType = typename WrappedClient::ParamDescType;
  using ParamSetViewType = typename WrappedClient::ParamSetViewType;
  using MessageSetType = typename WrappedClient::MessageSetType;
  using isModelObject = typename WrappedClient::isModelObject;

  using LookupTable =
      std::unordered_map<rt::string, std::weak_ptr<SharedClient>, std::hash<std::string_view>>;
  using ParamSetType =
      ParamAliasAdaptor<NRTClient, typename ParamDescType::ValueTuple>;

  using type = ClientPointer;

  constexpr static ParamDescType& getParameterDescriptors()
  {
    return WrappedClient::getParameterDescriptors();
  }
  constexpr static auto getMessageDescriptors()
  {
    return WrappedClient::getMessageDescriptors();
  }

  index          audioChannelsIn() const noexcept { return 0; }
  index          audioChannelsOut() const noexcept { return 0; }
  index          controlChannelsIn() const noexcept { return 0; }
  ControlChannel controlChannelsOut() const noexcept { return {0, 0}; }
  index          audioBuffersIn() const noexcept
  {
    return ParamDescType::template NumOf<InputBufferT>();
  }
  index audioBuffersOut() const noexcept
  {
    return ParamDescType::template NumOf<BufferT>();
  }

  NRTSharedInstanceAdaptor(ParamSetType& p, FluidContext c) : mParams{p}
  {
    // Not using the nifty operator[] of unordered map, because it deault
    // constructs the value object, giving us shared_ptr<nullptr>
    rt::string name = p.template get<0>();
    if (!mClientTable.count(name)) // key not already in table
    {
      mClient = std::make_shared<SharedClient>(p.instance(), c);
      mClientTable.emplace(name, mClient);
    }
    else
      mClient = mClientTable[name].lock()->shared();
  }

  NRTSharedInstanceAdaptor(const NRTSharedInstanceAdaptor& x) { *this = x; }

  NRTSharedInstanceAdaptor& operator=(const NRTSharedInstanceAdaptor& x)
  {
    mParams = x.mParams;
    mClient = x.mClient;
    return *this;
  }

  ~NRTSharedInstanceAdaptor()
  {
    if (mClient &&
        mClient.use_count() == 1) // is this the last remaining user of this
                                  // Corpus, except the hash table?
      mClientTable.erase(
          mParams.template get<0>()); // then remove it from the universe
  }

  static ClientPointer lookup(rt::string name)
  {
    return mClientTable.count(name) ? mClientTable[name].lock()->shared()
                                    : ClientPointer{};
  }

  template <size_t N, typename T, typename... Args>
  decltype(auto) invoke(T&, Args&&... args)
  {
    mProcessParams = mParams.instance();
    mClient->setParams(mProcessParams);
    return WrappedClient::getMessageDescriptors().template invoke<N>(
        *mClient, std::forward<Args>(args)...);
    // return mClient->template
  }

  template <typename T>
  Result process(FluidContext& c)
  {
    mProcessParams = mParams.instance();
    mClient->setParams(mProcessParams);
    return mClient->template process<T>(c);
  }

  void setParams(ParamSetType&) {}

private:
  ParamSetType                         mParams;
  ClientPointer                        mClient;
  typename WrappedClient::ParamSetType mProcessParams{
      NRTClient::getParameterDescriptors(), FluidDefaultAllocator()};
  static LookupTable mClientTable;
};

template <typename NRTClient> // init lookup table
typename NRTSharedInstanceAdaptor<NRTClient>::LookupTable
    NRTSharedInstanceAdaptor<NRTClient>::mClientTable{};


} // namespace client
} // namespace fluid
