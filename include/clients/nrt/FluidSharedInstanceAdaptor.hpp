#pragma once

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterSet.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>

#include <string>
#include <functional>
#include <unordered_map>

namespace fluid {
namespace client {

template<typename, typename> class ParamAliasAdaptor;

template<typename NRTClient, typename...Ts>
class ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>
{
  using WrappedClient = ClientWrapper<NRTClient>;
  using ParamSetType = typename WrappedClient::ParamSetType;
  using ListenersArray = std::array<std::vector<std::function<void()>>,sizeof...(Ts)>;
  
  struct ListeningParams
  {
    ParamSetType params{ClientWrapper<NRTClient>::getParameterDescriptors()};
    ListenersArray listeners;
  };

  using ParamsPointer = std::shared_ptr<ListeningParams>;
  using LookupTable = std::unordered_map<std::string,ParamsPointer>;
  
  template <size_t N>
  using ParamType = typename WrappedClient::ParamDescType::template ParamType<N>;

public:

  ParamAliasAdaptor(typename NRTClient::ParamDescType&):mParams{new ListeningParams()}
  {}
  
  ~ParamAliasAdaptor()
  {
    if(mParams && mParams.use_count() == 2) //is this the last remaining user of this Corpus, except the hash table?
    {
      auto name = mParams->params.template get<0>();
      mParamsTable.erase(name); //then remove it from the universe
    }
  }

  typename WrappedClient::ParamSetType& instance()
  {
    return mParams->params;
  }

  auto keepConstrained(bool keep)
  {
    return mParams->params.keepConstrained(keep);
  }

  std::array<Result, sizeof...(Ts)> constrainParameterValues()
  {
    return mParams->params.constrainParameterValues();
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setParameterValues(bool reportage, Args &&... args)
  {
    return mParams->params.template setParameterValues<Func>(reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setFixedParameterValues(bool reportage, Args &&... args)
  {
    auto results = mParams->params.template setFixedParameterValues<Func>(reportage, std::forward<Args>(args)...);
    
    std::string name = mParams->params.template get<0>();
    
    if(!name.size())
    {
      results[0].addMessage("Shared object given no name – won't be shared!");
      results[0].set(Result::Status::kWarning);
      return results;
    }
    
    if(!mParamsTable.count(name))//key not already in table
    {
      mParamsTable.emplace(name, mParams);
    }
    
    mParams = mParamsTable[name];
    return results;
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setMutableParameterValues(bool reportage, Args &&... args)
  {
    return mParams->params.template setMutableParameterValues<Func>(reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void forEachParam(Args &&... args)
  {
     mParams->params.template forEachParam<Func>(std::forward<Args>(args)...);
  }

  template <typename T, template <size_t, typename> class Func, typename... Args>
  void forEachParamType(Args &&... args)
  {
    mParams->params.template forEachParamType<T,Func>(std::forward<Args>(args)...);
  }

  void reset() { mParams->params.reset(); }

  template <size_t N>
  void set(typename ParamType<N>::type &&x, Result *reportage) noexcept
  {
    mParams->params.template set<N>(std::forward<typename ParamType<N>::type>(x), reportage);

    auto listeners = mParams->listeners[N];
    for(auto&& l:listeners) l();
  }

  template <std::size_t N>
  auto &get() const
  {
    return mParams->params.template get<N>();
  }

  template<size_t offset>
  auto subset()
  {
    return mParams->params.template subset<offset>();
  }

  template<size_t N, typename F>
  void addListener(F&& f){
     mParams->listeners[N].emplace_back(std::forward<F>(f));
  }

  private:
    ParamsPointer mParams;
    static LookupTable mParamsTable;
//    std::array<std::vector<std::function<void()>>,sizeof...(Ts)> mListeners;
};

//template<typename NRTClient, typename...Ts> //init master param set
//typename ClientWrapper<NRTClient>::ParamSetType
//ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>::mParams{ClientWrapper<NRTClient>::getParameterDescriptors()};

//template<typename NRTClient, typename...Ts> //init parameter listeners
//std::array<std::vector<std::function<void()>>,sizeof...(Ts)>
//ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>::mListeners{};

template<typename NRTClient, typename...Ts>
typename ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>::LookupTable
ParamAliasAdaptor<NRTClient, std::tuple<Ts...>>::mParamsTable{};


template<typename NRTClient>
class NRTSharedInstanceAdaptor : public OfflineIn, public OfflineOut
{

public:
  using WrappedClient = ClientWrapper<NRTClient>;
  using ClientPointer = typename std::shared_ptr<NRTClient>;
  using ClientWeakPointer = typename std::weak_ptr<const NRTClient>;
  using ParamDescType = typename WrappedClient::ParamDescType;
  using ParamSetViewType = typename WrappedClient::ParamSetViewType;
  using MessageSetType = typename WrappedClient::MessageSetType;
  using LookupTable = std::unordered_map<std::string,ClientPointer>;
  using ParamSetType =  ParamAliasAdaptor<NRTClient, typename ParamDescType::ValueTuple>;

  using type = ClientPointer;

  constexpr static ParamDescType getParameterDescriptors() { return NRTClient::getParameterDescriptors(); }
  constexpr static auto getMessageDescriptors() { return WrappedClient::getMessageDescriptors();}

  size_t audioChannelsIn()    const noexcept { return 0; }
  size_t audioChannelsOut()   const noexcept { return 0; }
  size_t controlChannelsIn()  const noexcept { return 0; }
  size_t controlChannelsOut() const noexcept { return 0; }
  size_t audioBuffersIn()  const noexcept { return ParamDescType:: template NumOf<InputBufferT>();   }
  size_t audioBuffersOut() const noexcept { return ParamDescType:: template NumOf<BufferT>();  }

  NRTSharedInstanceAdaptor(ParamSetType& p): mParams{p}
  {
      //Not using the nifty operator[] of unordered map, because it deault
    //constructs the value object, giving us shared_ptr<nullptr>
    std::string name = p.template get<0>();
    if(!mClientTable.count(name)) //key not already in table
       mClientTable.emplace(name, ClientPointer(new NRTClient(p.instance())));
    mClient = mClientTable[name];
  }

  ~NRTSharedInstanceAdaptor()
  {
    if(mClient && mClient.use_count() == 2) //is this the last remaining user of this Corpus, except the hash table?
      mClientTable.erase(mParams.get().instance().template get<0>()); //then remove it from the universe
  }


  static ClientPointer lookup(std::string name)
  {
    return mClientTable.count(name) ? (mClientTable)[name] : ClientPointer{};
  }

  template<size_t N, typename T, typename...Args>
  decltype(auto) invoke(T&, Args&&...args)
  {
    mProcessParams = mParams.get().instance();
    mClient->setParams(mProcessParams);
    return WrappedClient::getMessageDescriptors().template invoke<N>(*mClient, std::forward<Args>(args)...);
    //return mClient->template
  }

  template<typename T>
  Result process(FluidContext& c)
  {
    mProcessParams = mParams.get().instance();
    mClient->setParams(mProcessParams);
    return mClient->template process<T>(c);
  }

  void setParams(ParamSetType&) {}

private:
  std::reference_wrapper<ParamSetType> mParams;
  ClientPointer mClient;
  typename WrappedClient::ParamSetType mProcessParams{NRTClient::getParameterDescriptors()};
  static LookupTable mClientTable;
};

template<typename NRTClient> //init lookup table
typename NRTSharedInstanceAdaptor<NRTClient>::LookupTable NRTSharedInstanceAdaptor<NRTClient>::mClientTable{};




} //client
} //fluid
