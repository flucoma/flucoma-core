#pragma once

#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"
#include "../../algorithms/public/OnsetSegmentation.hpp"
#include "../../data/TensorTypes.hpp"

#include "../common/MessageSet.hpp"

#include <string>
#include <tuple>

namespace fluid {
namespace client {


auto constexpr MessageTestParams = defineParameters();

struct ReturnSetOfStrings
{
  template<typename Client>
  MessageResult<FluidTensor<std::string,1>> operator()(Client& c)
  { return c.doStrings(); }
};

struct ReturnSetOfNumbers
{
  template<typename Client>
  MessageResult<FluidTensor<double,1>> operator()(Client& c)
  { return c.doNumbers(); }
};

struct ReturnString
{
  template<typename Client>
  MessageResult<std::string> operator()(Client& c)
  { return c.doOneString(); }
};

struct ReturnIntegralType
{
  template<typename Client>
  MessageResult<intptr_t> operator()(Client& c)
  { return c.doOneNumber(); }
};

struct ReturnBufferSize
{
  template<typename Client>
  MessageResult<intptr_t> operator()(Client& c, std::shared_ptr<BufferAdaptor> buf)
  { return c.doBuffer(buf); }
};

struct ReceiveStringAndNumbers
{
  template<typename Client>
  MessageResult<void> operator()(Client& client, std::string labelName, double a, double b, double c)
  { return client.doTakeString(labelName, a, b, c); }
};

auto constexpr MessageTestMessages = defineMessages(
  Message<ReturnSetOfStrings>("testReturnStrings"),
  Message<ReturnSetOfNumbers>("testReturnNumbers"),
  Message<ReturnString>("testReturnOneString"),
  Message<ReturnIntegralType>("testReturnOneNumber"),
  Message<ReturnBufferSize>("testAccessBuffer"),
  Message<ReceiveStringAndNumbers>("testPassString")
);

class MessageTest : public FluidBaseClient<decltype(MessageTestParams), MessageTestParams, decltype(MessageTestMessages),MessageTestMessages>,
                   public OfflineIn,
                   public OfflineOut
{
public:

  MessageTest(ParamSetViewType &p) : FluidBaseClient(p)
  {}

  template <typename T>
  Result process(FluidContext& c) { return {}; }
  
  MessageResult<FluidTensor<std::string,1>> doStrings()
  {
    return FluidTensor<std::string,1>{"Hello","I","Love","you"};
  }

  MessageResult<FluidTensor<double,1>> doNumbers()
  {
    auto result = FluidTensor<double,1>(100);
    std::iota(result.begin(),result.end(), 0.0);
    return result;
  }
  
  MessageResult<std::string> doOneString()
  {
    return std::string("Hello I Love you");
  }

  MessageResult<intptr_t> doOneNumber()
  {
    return 12345;
  }

  MessageResult<intptr_t> doBuffer(std::shared_ptr<BufferAdaptor> b)
  {
    if(!b) return  {Result::Status::kError, "Null passed"};
    auto buf = BufferAdaptor::Access(b.get());
    if(!buf.exists()) return {Result::Status::kError, "No buffer found"};
    return buf.numFrames();
  }
 
  MessageResult<void> doTakeString(std::string s, double a, double b, double c)
  {
    std::cout << "Received " << s << ' ' << a << ' ' << b << ' ' << c << '\n';
    return {};
  }
};

using NRTThreadingMessageTest = NRTThreadingAdaptor<MessageTest>;

} // namespace client
} // namespace fluid
