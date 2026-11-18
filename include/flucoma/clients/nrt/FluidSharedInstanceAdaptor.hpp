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
#include <algorithm>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fluid {
namespace client {

template <typename>
class NRTSharedInstanceAdaptor;

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
    rt::string     name; // key in the lookup tables; shared by all aliases

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
      : mBinding{std::make_shared<Binding>()}
  {
    mBinding->data = std::make_shared<ListeningParams>();
  }

  ~ParamAliasAdaptor()
  {
    // Copies of a ParamAliasAdaptor share one Binding: only the last copy of
    // this object, if it is also the last user of the shared data (except the
    // hash table), removes the data from the universe
    if (mBinding && mBinding.use_count() == 1 && data() &&
        data().use_count() == 1)
      eraseEntryFor(data());
  }

  typename WrappedClient::ParamSetType& instance() { return data()->params; }

  auto keepConstrained(bool keep)
  {
    return data()->params.keepConstrained(keep);
  }

  std::array<Result, sizeof...(Ts)> constrainParameterValues()
  {
    return data()->params.constrainParameterValues();
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setParameterValues(bool reportage,
                                                       Args&&... args)
  {
    return data()->params.template setParameterValues<Func>(
        reportage, std::forward<Args>(args)...);
  }


  void constrainParameterValuesRT(std::array<Result, sizeof...(Ts)>* results)
  {
    data()->params.constrainParameterValuesRT(results);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void setParameterValuesRT(std::array<Result, sizeof...(Ts)>* reportage,
                            Args&&... args)
  {
    data()->params.template setParameterValuesRT<Func>(
        reportage, std::forward<Args>(args)...);
  }


  Result lookup(rt::string name)
  {
    return find(name) ? Result{}
                      : Result{Result::Status::kWarning, name, " not found"};
  }

  /// Point this object (and every copy of it, e.g. the ones held by the
  /// threading and sharing adaptors) at the data registered under name.
  /// Listeners this object registered move with it, and are then fired so
  /// that hosts can refresh the parameter values they display.
  Result refer(rt::string name)
  {
    ParamsPointer target = find(name);
    if (!target) return {Result::Status::kWarning, name, " not found"};
    if (target == data()) return {};

    ParamsPointer old = data();
    std::vector<ListenerEntry> moved = moveListeners(*old, *target);
    data() = target;
    // were we the last user of the old data? then unregister it
    if (old.use_count() == 1) eraseEntryFor(old);

    for (auto&& l : moved) l.first();
    return {};
  }

  /// The name this object's shared data is registered under
  const rt::string& name() const { return data()->name; }

  /// Give this object its name at creation time: if the name is already in
  /// use, start sharing that entry, otherwise register a new one.
  Result setName(rt::string name)
  {
    if (!name.size())
      return {Result::Status::kWarning,
              "Shared object given no name – won't be shared!"};

    if (!find(name)) // key not already in table
    {
      data()->name = name;
      mParamsTable.emplace(name, data());
    }

    return refer(name);
  }

  /// Change the name the shared data is registered under. The data and
  /// parameters are untouched, and every object sharing them sees the new
  /// name. Fails if the new name is empty or already in use.
  Result rename(rt::string newName)
  {
    if (!newName.size())
      return {Result::Status::kError, "Can't rename: no name given"};

    if (newName == data()->name) return {};

    if (find(newName))
      return {Result::Status::kError, "Can't rename: ", newName,
              " already exists"};

    rt::string oldName = data()->name;
    auto       entry = mParamsTable.find(oldName);

    if (entry != mParamsTable.end() && entry->second.lock() == data())
    {
      auto node = mParamsTable.extract(entry);
      node.key() = newName;
      mParamsTable.insert(std::move(node));
    }
    else // was unnamed (and hence unshared) until now
      mParamsTable.emplace(newName, data());

    data()->name = newName;
    NRTSharedInstanceAdaptor<NRTClient>::rekey(oldName, newName);
    return {};
  }

  template<typename Func,typename... Args>
  auto setPrimaryParameterValues(bool reportage,Func&& f, Args&&...args)
  {
     return data()->params.setPrimaryParameterValues(reportage, std::forward<Func>(f),std::forward<Args>(args)...);
  }
  
  template <template <size_t N, typename T> class Func, typename... Args>
  auto setFixedParameterValues(bool reportage, Args&&... args)
  {
    return data()->params.template setFixedParameterValues<Func>(
        reportage, std::forward<Args>(args)...);
  }

  template<size_t N> 
  typename ParamType<N>::type applyConstraintsTo(typename ParamType<N>::type x) const
  {
      return data()->template applyConstraintsTo<N>(x); 
  }

  template <size_t N>
  typename ParamType<N>::type
  applyConstraintToMax(typename ParamType<N>::type x) const
  {
    return data()->template applyConstraintToMax<N>(x);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setMutableParameterValues(bool reportage,
                                                              Args&&... args)
  {
    return data()->params.template setMutableParameterValues<Func>(
        reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void forEachParam(Args&&... args)
  {
    data()->params.template forEachParam<Func>(std::forward<Args>(args)...);
  }

  template <typename T, template <size_t, typename> class Func,
            typename... Args>
  void forEachParamType(Args&&... args)
  {
    data()->params.template forEachParamType<T, Func>(
        std::forward<Args>(args)...);
  }


  //lambda version
  template <typename T, class Func,
            typename... Args>
  void forEachParamType(Func&& f, Args&&... args)
  {
    data()->params.template forEachParamType<T>(std::forward<Func>(f),std::forward<Args>(args)...);
  }
  
  void reset() { data()->params.reset(); }

  template <size_t N>
  void set(typename ParamType<N>::type&& x, Result* reportage) noexcept
  {
    data()->params.template set<N>(
        std::forward<typename ParamType<N>::type>(x), reportage);

    auto listeners = data()->listeners[N];
    for (auto&& l : listeners) l.first();
  }

  template <std::size_t N>
  auto& get() const
  {
    return data()->params.template get<N>();
  }

  template <size_t offset>
  auto subset()
  {
    return data()->params.template subset<offset>();
  }
  
  template <size_t N>
  auto descriptorAt()
  {
    return data()->params.template descriptor<N>();
  }
  
  template <typename Tuple>
  void fromTuple(Tuple const& vals)
  {
    data()->params.fromTuple(vals);

    for (auto&& listeners : data()->listeners)
      for (auto&& l : listeners) l.first();
  }

  typename ParamSetType::ValueTuple toTuple()
  {
    return {data()->params.toTuple()};
  }

  template <size_t N, typename F>
  void addListener(F&& f, void* key)
  {
    data()->listeners[N].emplace_back(ListenerEntry{std::forward<F>(f), key});
    auto& keys = mBinding->listenerKeys;
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
      keys.push_back(key);
  }

  template <size_t N>
  void removeListener(void* key)
  {
    auto& listeners = data()->listeners[N];
    listeners.erase(
        std::remove_if(listeners.begin(), listeners.end(),
                       [&key](ListenerEntry& e) { return e.second == key; }),
        listeners.end());
  }

private:
  /// What a host object is currently pointing at. Shared by all copies of
  /// this ParamAliasAdaptor, so that refer() reaches every one of them.
  struct Binding
  {
    ParamsPointer      data;
    std::vector<void*> listenerKeys; // listeners this object has added
  };

  ParamsPointer& data() const { return mBinding->data; }

  /// Live entry for name, if any (pruning an expired one)
  static ParamsPointer find(const rt::string& name)
  {
    auto entry = mParamsTable.find(name);
    if (entry == mParamsTable.end()) return {};
    ParamsPointer p = entry->second.lock();
    if (!p) mParamsTable.erase(entry);
    return p;
  }

  /// Remove the table entry for p, if the entry really is p
  static void eraseEntryFor(const ParamsPointer& p)
  {
    auto entry = mParamsTable.find(p->name);
    if (entry != mParamsTable.end() && entry->second.lock() == p)
      mParamsTable.erase(entry);
  }

  /// Move this object's listeners from one set of shared data to another
  std::vector<ListenerEntry> moveListeners(ListeningParams& from,
                                           ListeningParams& to)
  {
    auto&                      keys = mBinding->listenerKeys;
    std::vector<ListenerEntry> moved;
    for (size_t i = 0; i < from.listeners.size(); ++i)
    {
      auto& src = from.listeners[i];
      auto  ours = std::stable_partition(
          src.begin(), src.end(), [&keys](ListenerEntry const& e) {
            return std::find(keys.begin(), keys.end(), e.second) == keys.end();
          });
      for (auto it = ours; it != src.end(); ++it)
      {
        to.listeners[i].push_back(*it);
        moved.push_back(*it);
      }
      src.erase(ours, src.end());
    }
    return moved;
  }

  std::shared_ptr<Binding> mBinding;
  static LookupTable       mParamsTable;
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
  using isModelObject = typename WrappedClient::isModelObject;

private:
  // Messages every shared object answers, on top of its client's own. They
  // act on this adaptor (i.e. on the name and the lookup tables), not on the
  // client, so they are declared here once for all shared clients.
  struct ReferOp
  {
    template <typename A>
    MessageResult<void> operator()(A& a, std::string name) const
    {
      return a.refer(std::move(name));
    }
  };

  struct SetNameOp
  {
    template <typename A>
    MessageResult<void> operator()(A& a, std::string name) const
    {
      return a.setname(std::move(name));
    }
  };

  struct GetNameOp
  {
    template <typename A>
    MessageResult<std::string> operator()(A& a) const
    {
      return a.getname();
    }
  };

  using ClientMessages = MessageTuple_t<typename WrappedClient::MessageSetType>;
  using NamingMessages = std::tuple<
      Message<ReferOp, MessageResult<void>, NRTSharedInstanceAdaptor,
              std::string>,
      Message<SetNameOp, MessageResult<void>, NRTSharedInstanceAdaptor,
              std::string>,
      Message<GetNameOp, MessageResult<std::string>,
              NRTSharedInstanceAdaptor>>;

public:
  /// How many of this object's messages come from the wrapped client; the
  /// naming messages (refer, setname, getname) follow them
  static constexpr size_t NumClientMessages =
      std::tuple_size<ClientMessages>::value;

  using MessageSetType = MessageSet<decltype(std::tuple_cat(
      std::declval<ClientMessages>(), std::declval<NamingMessages>()))>;

  using LookupTable =
      std::unordered_map<rt::string, std::weak_ptr<SharedClient>, std::hash<std::string_view>>;
  using ParamSetType =
      ParamAliasAdaptor<NRTClient, typename ParamDescType::ValueTuple>;

  using type = ClientPointer;

  constexpr static ParamDescType& getParameterDescriptors()
  {
    return WrappedClient::getParameterDescriptors();
  }
  static MessageSetType getMessageDescriptors()
  {
    return MessageSetType{std::tuple_cat(
        WrappedClient::getMessageDescriptors().messages(),
        NamingMessages{{"refer", ReferOp{}},
                       {"setname", SetNameOp{}},
                       {"getname", GetNameOp{}}})};
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
    rt::string name = p.name();
    if (!mClientTable.count(name)) // key not already in table
    {
      mClient = std::make_shared<SharedClient>(p.instance(), c);
      setClientName(*mClient, name);
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
      mClientTable.erase(mParams.name()); // then remove it from the universe
  }

  /// Move the entry for oldName to newName, leaving the client it points to
  /// untouched. Called by ParamAliasAdaptor::rename(); does nothing if there
  /// is no client registered under oldName yet.
  static void rekey(const rt::string& oldName, const rt::string& newName)
  {
    auto entry = mClientTable.find(oldName);
    if (entry == mClientTable.end()) return;
    auto node = mClientTable.extract(entry);
    node.key() = newName;
    auto inserted = mClientTable.insert(std::move(node));
    if (!inserted.inserted) return;
    if (auto client = inserted.position->second.lock())
      setClientName(*client, newName);
  }

  static ClientPointer lookup(rt::string name)
  {
    return mClientTable.count(name) ? mClientTable[name].lock()->shared()
                                    : ClientPointer{};
  }

  template <size_t N, typename T, typename... Args>
  decltype(auto) invoke(T&, Args&&... args)
  {
    if constexpr (N < NumClientMessages)
    {
      mProcessParams = mParams.instance();
      mClient->setParams(mProcessParams);
      return WrappedClient::getMessageDescriptors().template invoke<N>(
          *mClient, std::forward<Args>(args)...);
    }
    else // one of the naming messages, which act on this adaptor
      return getMessageDescriptors().template invoke<N>(
          *this, std::forward<Args>(args)...);
  }

  /// Message: share the data (and parameters) of the object called name
  /// from now on
  MessageResult<void> refer(std::string name)
  {
    rt::string newName(name.c_str());
    if (newName == mParams.name()) return {};

    rt::string oldName = mParams.name();
    Result     r = mParams.refer(newName);
    // an error, not a warning: hosts carry on (and report success) after
    // warnings, but here nothing has happened
    if (!r.ok()) return {Result::Status::kError, r.message()};

    ClientPointer target = lookup(newName);
    if (!target) // the data has no client yet: make one, as at construction
    {
      FluidContext c;
      target = std::make_shared<SharedClient>(mParams.instance(), c);
      setClientName(*target, newName);
      mClientTable.emplace(newName, target);
    }
    // were we the last user of the old client? then unregister it
    if (mClient && mClient.use_count() == 1)
    {
      auto entry = mClientTable.find(oldName);
      if (entry != mClientTable.end() && entry->second.lock() == mClient)
        mClientTable.erase(entry);
    }
    mClient = target;
    return {};
  }

  /// Message: change the name the shared data is registered under, for every
  /// object sharing it
  MessageResult<void> setname(std::string name)
  {
    Result r = mParams.rename(rt::string(name.c_str()));
    if (!r.ok()) return {r.status(), r.message()};
    return {};
  }

  /// Message: the name the shared data is registered under
  MessageResult<std::string> getname()
  {
    return std::string(mParams.name().data(), mParams.name().size());
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
  // Clients that derive from NamedObject get told what they are called
  static void setClientName(SharedClient& client, const rt::string& name)
  {
    if constexpr (std::is_base_of<NamedObject, NRTClient>::value)
      client.setName(std::string_view(name.data(), name.size()));
  }

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
