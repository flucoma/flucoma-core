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

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fluid {
namespace client {

class Result
{
public:
  enum class Status { kOk, kWarning, kError, kCancelled };

  Result(Status s, std::string msg) : mStatus(s), mMsg(msg) {}

  template <typename... Args>
  Result(Status s, Args... args) : mStatus(s)
  {
    (void) std::initializer_list<int>{(mMsg << args, 0)...};
  }

  Result() = default;
  Result(const Result& x) : mStatus(x.mStatus), mMsg(x.mMsg.str()) {}

  Result& operator=(const Result& r)
  {
    if (&r == this) return *this;

    mStatus = r.mStatus;
    mMsg = std::ostringstream(r.mMsg.str());
    return *this;
  };

  Result(Result&& x) noexcept { *this = std::move(x); }
  Result& operator=(Result&& x) noexcept
  {
    if (this != &x)
    {
      using std::swap;
      mMsg = std::move(x.mMsg);
      swap(mStatus, x.mStatus);
    }
    return *this;
  };

  bool ok() const noexcept { return (mStatus == Status::kOk); }

  Status status() { return mStatus; }

  void set(Status r) noexcept { mStatus = r; }

  std::string message() const noexcept { return mMsg.str(); }

  template <typename... Ts>
  void addMessage(Ts... args)
  {
    (void) std::initializer_list<int>{(mMsg << args, 0)...};
  }

  void reset()
  {
    mMsg = std::ostringstream();
    mStatus = Status::kOk;
  }

private:
  Status            mStatus = Status::kOk;
  std::ostringstream mMsg;
};


template <typename T>
class MessageResult : public Result
{
public:
  using type = T;
  using Result::Result;
  MessageResult(T data) : mData{data} {}
  MessageResult() = default;
  operator T() const { return mData; }
  T& value() { return mData; }
  const T& value() const { return mData; }
private:
  T    mData;
  bool hasData;
};

template <>
class MessageResult<void> : public Result
{
public:
  using Result::Result;
};


class MessageList
{
public:
  template <typename... Args>
  void addWarn(Args&&... args)
  {
    add<Result::Status::kWarning>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void addError(Args&&... args)
  {
    add<Result::Status::kError>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void addInfo(Args&&... args)
  {
    add<Result::Status::kOk>(std::forward<Args>(args)...);
  }

  auto release()
  {
    auto res = std::vector<Result>{};
    std::swap(messages, res);
    return res;
  }

private:
  template <Result::Status S, typename... Args>
  void add(Args&&... args)
  {
    messages.emplace_back(S, std::forward<Args>(args)...);
  }

  std::vector<Result> messages;
};
} // namespace client
} // namespace fluid
