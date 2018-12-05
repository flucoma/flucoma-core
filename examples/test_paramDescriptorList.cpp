#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/rt/GainClient.hpp>
#include <clients/rt/BaseSTFTClient.hpp>

//#include "clients/common/ParameterInstance.hpp"

void testPseudoEnvironment() {

  //  clients
}

namespace fluid {
namespace client {

namespace impl {
// This seems faffy, but comes from Herb Sutter's reccomendation for avoiding
// specialising functions http://www.gotw.ca/publications/mill17.htm.
// Specialisations are
// below the wrapper template, just to try and reudce clutter
template <typename Client, typename T, size_t N> struct SetterDispatchImpl;
template <typename Client, typename T, size_t N> struct GetterDispatchImpl;
} // namespace impl

template <typename Client, typename... Ts> class DummyWrapper {
public:
  DummyWrapper(){}

  static auto makeClass(const std::tuple<Ts...> params) {
    //    MaxClass_Base::makeClass<FluidMaxWrapper>(nameSpace,className);
    processParameters(params, std::index_sequence_for<Ts...>{});
    return DummyWrapper();
  }

  template<size_t N,typename T>
  void set(T v)
  {
      mClient.template setter<N>()(v);
//    setterDispatch<T, N>(mClient)
  }

private:

  Client mClient;
  

  template <typename T, size_t N> static void setupAttribute(T &attr) {
    declareAttr(attr);
    using AttrType = std::remove_reference_t<decltype(attr)>;
    auto f = getterDispatch<AttrType, N>;
    //      CLASS_ATTR_ACCESSORS(*getClassPointer<FluidMaxWrapper>(), attr.name,
    //      (getterDispatch<AttrType, N>), (setterDispatch<AttrType,N>));
  }

  template <size_t... Is>
  static void processParameters(std::tuple<Ts...> params,
                                std::index_sequence<Is...>) {
    (void)std::initializer_list<int>{
        (setupAttribute<Ts, Is>(std::get<Is>(params)), 0)...};
  }
  
  static void declareAttr(FloatT& t){};

  static void declareAttr(LongT& t) {}

  static void declareAttr(BufferT& t) {}

  static void declareAttr(EnumT& t) {}

  template <typename T, size_t N> static void getterDispatch(Client *x) {
    impl::GetterDispatchImpl<Client, std::remove_const_t<T>, N>::f(x);
  }

  template <typename T, size_t N> static void setterDispatch(Client *x) {
    impl::SetterDispatchImpl<Client, std::remove_const_t<T>, N>::f(x);
  }

  

};

namespace impl {
template <typename Client, size_t N>
struct SetterDispatchImpl<Client, FloatT, N> {
  static void f(Client *x, double y) { x->template setter<N>()(y); }
};

template <typename Client, size_t N>
struct SetterDispatchImpl<Client, LongT, N> {
  static void f(Client *x, long y) { x->template setter<N>()(y); }
};

template <typename Client, size_t N>
struct GetterDispatchImpl<Client, FloatT, N> {
  static void f(Client *x) { x->template get<N>(); }
};
template <typename Client, size_t N>
struct GetterDispatchImpl<Client, LongT, N> {
  static void f(Client *x) { x->template get<N>(); }
};

} // namespace impl

template <typename Client, typename... Ts>
auto makeWrapper(const std::tuple<Ts...> &params) {
  return DummyWrapper<Client, typename Ts::first_type...>::makeClass(ParameterDescriptors<Ts...>::get(params));
}

} // namespace client
} // namespace fluid

int main(int argc, char *argv[]) {
  using namespace fluid::client;
//  GainClient<double> g;
  DummyWrapper<BaseSTFTClient<double>, decltype(STFTParams)> c;
  //makeWrapper<BaseSTFTClient<double>>(STFTParams);
  c.set<0>(1023); 
  c.set<2>(8);
  
  
}

