#pragma once

#include "ParameterSet.hpp"
#include "ParameterTypes.hpp"
#include <tuple>
#include <functional>

namespace fluid {
namespace client {

template<typename L, typename Ret, typename T, typename...Args>
struct Message
{
  Message(const char* n, L l): name{n}, op(l) {}
  const char* name;
  L op;
  using ReturnType = Ret;
  using IndexList = std::index_sequence_for<Args...>;
  using ArgumentTypes = std::tuple<Args...>;
  using Client = T;
  
  template<size_t N>
  using ArgType = typename std::tuple_element<N, ArgumentTypes>::type;
  
  decltype(auto) operator()(Client& c, Args...args) const
  {
    return op(c,std::forward<Args>(args)...); 
    //(c.*op)(std::forward<Args>(args)...);
  }
  
  template<typename ArgType, template <size_t, typename> class Func, typename...ArgsToPass>
  static void forEachArg(ArgumentTypes& tuple, ArgsToPass&&...args)
  {
    using ArgIndices =typename impl::FilterTupleIndices<IsArgType<ArgType>,ArgumentTypes,IndexList>::type;
    forEachArgImpl<Func>(tuple, ArgIndices{}, std::forward<ArgsToPass>(args)...);
  }
  
  private:
  
  template <template <size_t, typename> class Func, typename... ArgsToPass, size_t... Is>
  static void forEachArgImpl(ArgumentTypes& tuple, std::index_sequence<Is...>, ArgsToPass&&... args)
  {
    (void)std::initializer_list<int>{
      (Func<Is, ArgType<Is>>()(std::get<Is>(tuple), std::forward<ArgsToPass>(args)...), 0)...
    };
  }
  
  
  template <typename ArgType>
  struct IsArgType
  {
    template <typename U>
    using apply = std::is_same<ArgType,U>;
  };
  
};

template<typename Ret, typename T, typename...Args>
auto makeMessage(const char* name, Ret (T::* pmf) (Args...)  )
{
  auto l  = [=](T& t, Args...args){ return (t.*pmf)(std::forward<Args>(args)...);};
  return Message<std::decay_t<decltype(l)>,Ret,T,Args...>{name, l};
}

template<typename Ret, typename T, typename...Args>
auto makeMessage(const char* name, Ret (T::* pmf) (Args...) const )
{
  auto l  = [=](T& t, Args...args){ return (t.*pmf)(std::forward<Args>(args)...);};
  return Message<std::decay_t<decltype(l)>,Ret,T,Args...>{name, l};
}

template <typename>
class MessageSet;

template <typename... Ts>
class MessageSet<std::tuple<Ts...>>
{
  using MessagesType = std::tuple<Ts...>;
  template<size_t N>
  using MessageTypeAt = typename std::tuple_element<N,MessagesType>::type;
public:
  
  template <size_t N>
  using MessageDescriptorAt = MessageTypeAt<N>;
  
  using IndexList = std::index_sequence_for<Ts...>;
  
  MessageSet(const Ts&&... ts) : mMessages{std::make_tuple(ts...)} {}
  MessageSet(const std::tuple<Ts...>&& t): mMessages{t} {}

  template<size_t N>
  std::string name() const { return std::get<N>(mMessages).name; }

  size_t size() const noexcept { return sizeof...(Ts); }
  
  template <template <size_t N, typename T> class Func>
  void iterate() const
  {
    iterateImpl<Func>(IndexList());
  }
  
  template <size_t N>
  auto& get() const
  {
    return std::get<0>(std::get<N>(mMessages));
  }

  template <size_t N,typename...Us>
  decltype(auto) invoke(Us&&...us) const {
    return std::get<N>(mMessages)(std::forward<Us>(us)...);
  }

private:
  MessagesType mMessages;

  template <template <size_t N, typename T> class Op, size_t... Is>
  void iterateImpl(std::index_sequence<Is...>) const
  {
    (void)std::initializer_list<int>{(Op<Is, MessageTypeAt<Is>>()(std::get<Is>(mMessages)), 0)...};
  }
};

template<typename...Args>
MessageSet<std::tuple<Args...>> defineMessages(Args&&...args){ return {std::forward<Args>(args)...}; }

//Boilerplate macro for clients
#define FLUID_DECLARE_MESSAGES(...) \
    using MessageType = std::add_const_t<decltype(defineMessages(__VA_ARGS__))>; \
    static MessageType getMessageDescriptors() { return defineMessages(__VA_ARGS__);}

auto NoMessages = defineMessages();

} //client
} //fluid

