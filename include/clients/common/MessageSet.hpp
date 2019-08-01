#pragma once

#include "ParameterSet.hpp"
#include "ParameterTypes.hpp"
#include <tuple>
#include <functional>

namespace fluid {
namespace client {

template<typename, typename, typename, typename>
struct MessageDescriptor;

namespace impl{
template <typename Client, typename Message>
struct MessageDescriptorBuilder
{
  using Functor = typename Message::type;
  //Scrape the types from the lambda / functor args (but not the template arg at the start)
  template<typename F, typename Ret, typename A, typename... Rest>
  static std::tuple<Rest...> args1ToN(Ret (F::*)(A, Rest...));

  template<typename F, typename Ret, typename A, typename... Rest>
  static std::tuple<Rest...> args1ToN(Ret (F::*)(A, Rest...) const);


  template<typename F, typename Ret, typename A, typename... Rest>
  static Ret returnType(Ret (F::*)(A, Rest...));

  template<typename F, typename Ret, typename A, typename... Rest>
  static Ret returnType(Ret (F::*)(A, Rest...) const);

  using ReturnType    = decltype(returnType(&Functor::template operator()<Client>));
  using ArgumentTuple = decltype(args1ToN(&Functor::template operator()<Client>));
  using type = MessageDescriptor<Client, Functor, ReturnType,ArgumentTuple>;
};
} //impl

//Takes functor or a lambda to execute
//The operator() of this *must* be generic / templated on the first argument
//(which will be a client l-value ref), the rest shouldn't be generic, but can
// be whatever we know how to parse from hosts
template<typename Client, typename Functor, typename Ret, typename...Args>
struct MessageDescriptor<Client, Functor, Ret, std::tuple<Args...>>
{
  using ReturnType    = Ret;
  using ArgumentTypes = std::tuple<Args...>;
  using IndexList = std::make_index_sequence<std::tuple_size<ArgumentTypes>::value>;
  
  constexpr MessageDescriptor(const char* n, Functor lmbda): name{n},op{lmbda}
  {}
  
  decltype(auto) operator()(Client& c, Args&&...args) const
  {
    return op(c, std::forward<Args>(args)...);
  }
  
  const  Functor op;
  const char* name;
};

template<typename F>
struct Message
{
  constexpr Message(const char* n): name{n} {}
  const char* name;
  using type = F;
};

template<typename F>
constexpr auto makeMessage(const char* name, F f) { return Message<F>{name, f}; }

template <typename>
class MessageSet;

template <typename... Ts>
class MessageSet<std::tuple<Ts...>>
{
  using MessagesType = std::tuple<Ts...>;
  template<size_t N>
  using MessageTypeAt = typename std::tuple_element<N,MessagesType>::type;
public:
  
  template <typename Client, size_t N>
  using MessageDescriptorAt = typename impl::MessageDescriptorBuilder<Client,typename std::tuple_element<N,MessagesType>::type>::type;
  
  using IndexList = std::index_sequence_for<Ts...>;
  
  constexpr MessageSet(const Ts&&... ts) : mMessages{std::make_tuple(ts...)} {}
  constexpr MessageSet(const std::tuple<Ts...>&& t): mMessages{t} {}
//  constexpr MessageSet(){}

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
    using Op = typename MessageTypeAt<N>::type;
    return Op()(std::forward<Us>(us)...) ;
  }

private:
  MessagesType mMessages;

  template <template <size_t N, typename T> class Op, size_t... Is>
  void iterateImpl(std::index_sequence<Is...>) const
  {
    std::initializer_list<int>{(Op<Is, MessageTypeAt<Is>>()(std::get<Is>(mMessages)), 0)...};
  }
};

template<typename...Args>
constexpr MessageSet<std::tuple<Args...>> defineMessages(Args&&...args){ return {std::forward<Args>(args)...}; }

auto constexpr NoMessages = defineMessages();


} //client
} //fluid

