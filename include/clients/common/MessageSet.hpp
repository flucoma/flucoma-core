#pragma once

#include "ParameterSet.hpp"
#include "ParameterTypes.hpp"
#include <tuple>
#include <functional>

namespace fluid {
namespace client {

template<typename Ret, typename T, typename...Args>
struct Message
{
  constexpr Message(const char* n, Ret (T::* pmf) (Args...)): name{n}, op(pmf) {}
  const char* name;
  Ret (T::* op) (Args...);
  using ReturnType = Ret;
  using IndexList = std::index_sequence_for<Args...>;
  using ArgumentTypes = std::tuple<Args...>;
  using Client = T;
  decltype(auto) operator()(Client& c, Args...args) const
  {
    return (c.*op)(std::forward<Args>(args)...);
  }
};

template<typename Ret, typename T, typename...Args>
constexpr auto makeMessage(const char* name, Ret (T::* pmf) (Args...)) { return Message<Ret,T,Args...>{name, pmf}; }

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
  
  constexpr MessageSet(const Ts&&... ts) : mMessages{std::make_tuple(ts...)} {}
  constexpr MessageSet(const std::tuple<Ts...>&& t): mMessages{t} {}

  template<size_t N>
  std::string name() const { return std::get<N>(mMessages).name; }

  constexpr size_t size() const noexcept { return sizeof...(Ts); }
  
  template <template <size_t N, typename T> class Func>
  void iterate() const
  {
    iterateImpl<Func>(IndexList());
  }
  
  template <size_t N>
  constexpr auto& get() const
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
constexpr MessageSet<std::tuple<Args...>> defineMessages(Args&&...args){ return {std::forward<Args>(args)...}; }

//Boilerplate macro for clients
#define FLUID_DECLARE_MESSAGES(...) \
    using MessageType = std::add_const_t<decltype(defineMessages(__VA_ARGS__))>; \
    static constexpr MessageType getMessageDescriptors() { return defineMessages(__VA_ARGS__);}

auto constexpr NoMessages = defineMessages();

} //client
} //fluid

