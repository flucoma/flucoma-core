#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterTypes.hpp>
//#include <clients/rt/GainClient.hpp>
//#include <clients/rt/BaseSTFTClient.hpp>
#include <clients/rt/NMFMatch.hpp>
#include <random>

//#include "clients/common/ParameterInstance.hpp"

void testPseudoEnvironment() {

  //  clients
}
template<typename...Args>
void whatHappened(Args&&...args)
{
  puts(__PRETTY_FUNCTION__);
  std::initializer_list<int>{(std::cout << args << '\t',0)...};
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


  void process(std::vector<fluid::FluidTensorView<double,1>>& in,std::vector<fluid::FluidTensorView<double,1>>& out)
  {
    mClient.process(in,out);
  }


  template<size_t N>
  auto get()
  {
   return  mClient.template get<N>();
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
        (setupAttribute<typename Ts::first_type, Is>(std::get<Is>(params).first), 0)...};
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
  return DummyWrapper<Client, typename Ts::first_type...>::makeClass(params);
}


template<typename T>
struct Fetcher;

template<>
struct Fetcher<LongT>
{
  static size_t value(long* a){return *a;}
};


void fun(size_t a, size_t b)
{
  puts(__PRETTY_FUNCTION__);
  std::cout << a << '\t' << b;
}

template<typename Client>
struct ClientFactory
{
  static Client create(long ac, long* av)
  {
    using FixedParamIndices = typename Client::FixedParams;
    using ArgIndices = std::make_index_sequence<Client::FixedParams::size()>;
  
  
    return createImpl(ac, av, FixedParamIndices{},ArgIndices{});
  }
  
private:
  template<size_t...Is, size_t...Js>
  static auto createImpl(long ac, long* av, std::index_sequence<Is...>, std::index_sequence<Js...>)
  {
//    puts(__PRETTY_FUNCTION__);
//    fun(Fetcher< typename Client::template ParamDescriptorTypeAt<Is> >::value(av + Js)...);
    Client a{Fetcher< typename Client::template ParamDescriptorTypeAt<Is> >::value(av + Js)...};
     return a;
  }
  
};


} // namespace client
} // namespace fluid




int main(int argc, char *argv[]) {
  using namespace fluid::client;
  
  long a[2]{3,4};
  
  NMFMatch<double> b = (ClientFactory<NMFMatch<double>>::create(2, a));
  
//  whatHappened(NMFMatch<double>::AdjustableParams()); 
//  
//  whatHappened(NMFMatch<double>::FixedParams());
 
//  GainClient<double> g;
//  DummyWrapper<BaseSTFTClient<double>, decltype(STFTParams)> c;
  //makeWrapper<BaseSTFTClient<double>>(STFTParams);
//  DummyWrapper<TransientClient<double>, decltype(TransientParams)> c;
//  c.set<0>(100);
//  c.set<1>(512);
//  c.set<2>(1024);

//std::cout<<c.get<0>()<<'\n';
//c.set<0>(14);
//std::cout<<c.get<0>()<<'\n';


//  std::default_random_engine g;
//  std::uniform_real_distribution<double> noise(-1,1);
//
//  fluid::FluidTensor<double,1> i(1024);
//
//
//
//  fluid::FluidTensor<double,1> o(1024);
//  std::vector<fluid::FluidTensorView<double,1>> input{i};
//  std::vector<fluid::FluidTensorView<double,1>> output{o};
//
//  fluid::FluidTensor<double,1> collectInput(1024 * 256 + 1024);
//  fluid::FluidTensor<double,1> collectOutput((1024 * 256) + 1024);

  
//  for(int j = 0; j < 256; j++)
//  {
//    i.apply([&g,&noise](double& x){
//      x = noise(g);
//    });
//    collectInput(fluid::Slice(1024 +  (1024*j),1024)) = i;
//    c.process(input,output);
////    std::cout << *std::max_element(i.begin(), i.end()) << ' ' << *std::min_element(i.begin(), i.end()) << '\n';
//    collectOutput(fluid::Slice( 1024 * j,1024)) = o;
////                              auto i = output[0];
////      std::cout << *std::max_element(i.begin(), i.end()) << ' ' << *std::min_element(i.begin(), i.end()) << '\n';
//  }
//  std::cout << collectInput(fluid::Slice(1024,10));
//  std::cout << collectOutput(fluid::Slice(1024,10));
//     collectOutput.apply(collectInput,[](double& a, double& b){
//      a -= b;
//    });
//
////    std::cout << *std::max_element(collectInput.begin(), collectInput.end()) << '\n';
//
//  std::cout << *std::max_element(collectOutput.begin() + 512, collectOutput.begin() + (1024 * 256)) << '\n';

  
}

