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
      : mParams{std::make_shared<ListeningParams>()}
  {}

  ~ParamAliasAdaptor()
  {
    // copies share one binding: only the last copy, if it is also the last
    // user of the data (except the hash table), removes it from the universe
    if (mParams.lastCopy() && mParams.data().use_count() == 1)
      eraseEntryFor(mParams.data());
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
    return find(name) ? Result{}
                      : Result{Result::Status::kWarning, name, " not found"};
  }

  /// Point this object, and every copy of it, at the data registered under
  /// name. The listeners it added go along, and fire so that hosts can
  /// refresh what they display.
  Result refer(rt::string name)
  {
    ParamsPointer target = find(name);
    if (!target) return {Result::Status::kWarning, name, " not found"};
    if (target == mParams.data()) return {};

    ParamsPointer old = mParams.data();
    auto          moved = moveListeners(*old, *target);
    mParams.data() = target;
    // were we the last user of the old data? then remove it from the universe
    if (old.use_count() == 1) eraseEntryFor(old);
    for (auto&& l : moved) l.first();
    return {};
  }

  /// The name this object's shared data is registered under
  const rt::string& name() const { return mParams->name; }

  /// Give this object its name at creation time: if the name is already in
  /// use, start sharing that entry, otherwise register a new one.
  Result setName(rt::string name)
  {
    if (!name.size())
      return {Result::Status::kWarning,
              "Shared object given no name – won't be shared!"};

    if (!find(name)) // no live entry under this name
    {
      mParams->name = name;
      mParamsTable.insert_or_assign(name, mParams.data()); // over any dead one
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

    if (newName == mParams->name) return {};

    // a client that outlived its data (still held somewhere) keeps its name
    if (find(newName) || NRTSharedInstanceAdaptor<NRTClient>::lookup(newName))
      return {Result::Status::kError, "Can't rename: ", newName,
              " already exists"};

    mParamsTable.erase(newName); // a dead entry, if any, is in the way
    rt::string oldName = mParams->name;
    auto       entry = mParamsTable.find(oldName);

    if (entry != mParamsTable.end() && entry->second.lock() == mParams.data())
    {
      auto node = mParamsTable.extract(entry);
      node.key() = newName;
      mParamsTable.insert(std::move(node));
    }
    else // was unnamed (and hence unshared) until now
      mParamsTable.emplace(newName, mParams.data());

    mParams->name = newName;
    NRTSharedInstanceAdaptor<NRTClient>::rekey(oldName, newName);
    return {};
  }

  template<typename Func,typename... Args>
  auto setPrimaryParameterValues(bool reportage,Func&& f, Args&&...args)
  {
     return mParams->params.setPrimaryParameterValues(reportage, std::forward<Func>(f),std::forward<Args>(args)...);
  }
  
  template <template <size_t N, typename T> class Func, typename... Args>
  auto setFixedParameterValues(bool reportage, Args&&... args)
  {
    return mParams->params.template setFixedParameterValues<Func>(
        reportage, std::forward<Args>(args)...);
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
    return mParams->template applyConstraintToMax<N>(x);
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
    auto& keys = mParams.keys(); // so that refer() can take them along
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
      keys.push_back(key);
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
  /// The data registered under name, if it is still alive
  static ParamsPointer find(const rt::string& name)
  {
    auto entry = mParamsTable.find(name);
    return entry != mParamsTable.end() ? entry->second.lock() : ParamsPointer{};
  }

  /// Remove the table entry for p, if the entry really is p
  static void eraseEntryFor(const ParamsPointer& p)
  {
    auto entry = mParamsTable.find(p->name);
    if (entry != mParamsTable.end() && entry->second.lock() == p)
      mParamsTable.erase(entry);
  }

  /// What this object points at. All copies of a ParamAliasAdaptor (e.g.
  /// those the threading and sharing adaptors hold) share one Binding, so
  /// that refer() re-points them all; mParams-> reaches the shared data.
  class Binding
  {
  public:
    explicit Binding(ParamsPointer p)
        : mState{std::make_shared<State>(State{std::move(p), {}})}
    {}

    ParamsPointer&      operator->() const { return mState->data; }
    ParamsPointer&      data() const { return mState->data; }
    std::vector<void*>& keys() const { return mState->keys; }
    bool                lastCopy() const { return mState.use_count() == 1; }

  private:
    struct State
    {
      ParamsPointer      data;
      std::vector<void*> keys; // that this object added listeners with
    };

    std::shared_ptr<State> mState;
  };

  /// Move the listeners this object added from one set of data to another,
  /// returning them
  std::vector<ListenerEntry> moveListeners(ListeningParams& from,
                                           ListeningParams& to)
  {
    auto& keys = mParams.keys();
    auto  notOurs = [&keys](ListenerEntry const& e) {
      return std::find(keys.begin(), keys.end(), e.second) == keys.end();
    };
    std::vector<ListenerEntry> moved;
    for (size_t i = 0; i < from.listeners.size(); ++i)
    {
      auto& src = from.listeners[i];
      auto  ours = std::stable_partition(src.begin(), src.end(), notOurs);
      to.listeners[i].insert(to.listeners[i].end(), ours, src.end());
      moved.insert(moved.end(), ours, src.end());
      src.erase(ours, src.end());
    }
    return moved;
  }

  Binding            mParams;
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
  using isModelObject = typename WrappedClient::isModelObject;

  /// The client's own messages come first; refer, setname and getname follow
  static constexpr size_t NumClientMessages =
      WrappedClient::MessageSetType::IndexList::size();

private:
  // Messages every shared object answers, which act on this adaptor (the
  // name and the lookup tables) rather than on the client
  template <typename Ret, typename... Args>
  using NamingMessage = Message<Ret (*)(NRTSharedInstanceAdaptor&, Args...),
                                Ret, NRTSharedInstanceAdaptor, Args...>;
  using NamingMessages =
      std::tuple<NamingMessage<MessageResult<void>, std::string>,
                 NamingMessage<MessageResult<void>, std::string>,
                 NamingMessage<MessageResult<std::string>>>;

public:
  using MessageSetType = MessageSet<decltype(std::tuple_cat(
      std::declval<typename WrappedClient::MessageSetType const&>().messages(),
      std::declval<NamingMessages>()))>;

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
    return MessageSetType{std::tuple_cat(
        WrappedClient::getMessageDescriptors().messages(),
        NamingMessages{{"refer",
                        [](NRTSharedInstanceAdaptor& a, std::string name) {
                          return a.refer(std::move(name));
                        }},
                       {"setname",
                        [](NRTSharedInstanceAdaptor& a, std::string name) {
                          return a.setname(std::move(name));
                        }},
                       {"getname", [](NRTSharedInstanceAdaptor& a) {
                          return a.getname();
                        }}})};
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
    mClient = lookup(name);
    if (!mClient) // no live client under this name
    {
      mClient = std::make_shared<SharedClient>(p.instance(), c);
      setClientName(*mClient, name);
      mClientTable.insert_or_assign(name, mClient); // over any dead one
    }
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
      eraseEntryFor(mParams.name(), mClient); // then unregister it
  }

  /// Move the entry for oldName to newName, leaving the client it points to
  /// untouched. Called by ParamAliasAdaptor::rename(); does nothing if there
  /// is no client registered under oldName yet.
  static void rekey(const rt::string& oldName, const rt::string& newName)
  {
    // a dead entry under newName would block the move
    if (!lookup(newName)) mClientTable.erase(newName);
    auto entry = mClientTable.find(oldName);
    if (entry == mClientTable.end()) return;
    auto node = mClientTable.extract(entry);
    node.key() = newName;
    auto inserted = mClientTable.insert(std::move(node));
    if (!inserted.inserted) return;
    if (auto client = inserted.position->second.lock())
      setClientName(*client, newName);
  }

  /// The client registered under name, if it is still alive. Read-only, as
  /// real-time clients (e.g. KDTreeQuery) look clients up while processing.
  static ClientPointer lookup(rt::string name)
  {
    auto entry = mClientTable.find(name);
    return entry != mClientTable.end() ? entry->second.lock() : ClientPointer{};
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

  /// Message: share the data (and parameters) of the object called name from
  /// now on
  MessageResult<void> refer(std::string name)
  {
    rt::string newName(name.c_str());
    if (newName == mParams.name()) return {};

    // errors, not warnings: hosts carry on (and reply) after a warning, but
    // here nothing has happened
    ClientPointer target = lookup(newName);
    if (!target) return {Result::Status::kError, newName, " not found"};

    rt::string oldName = mParams.name();
    Result     r = mParams.refer(newName);
    if (!r.ok()) return {Result::Status::kError, r.message()};

    // were we the last user of the old client? then unregister it
    if (mClient.use_count() == 1) eraseEntryFor(oldName, mClient);
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

  /// Remove the table entry for name, if the entry really is c
  static void eraseEntryFor(const rt::string& name, const ClientPointer& c)
  {
    auto entry = mClientTable.find(name);
    if (entry != mClientTable.end() && entry->second.lock() == c)
      mClientTable.erase(entry);
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
