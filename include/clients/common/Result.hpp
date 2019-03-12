#pragma once

#include <vector>

namespace fluid {
namespace client {

class Result
{
public:
  enum class Status { kOk, kWarning, kError };

  Result(Status s, std::string msg)
      : mStatus(s)
      , mMsg(msg)
  {}

  template <typename... Args>
  Result(Status s, Args... args)
      : mStatus(s)
  {
    std::initializer_list<int>{(mMsg << args, 0)...};
  }

  Result() = default;
  Result(const Result &x)
      : mStatus(x.mStatus)
      , mMsg(x.mMsg.str())
  {}

  Result& operator=(const Result &r)
  {
    if(&r == this) return *this;
    
    mStatus = r.mStatus;
    mMsg = std::stringstream(r.mMsg.str());
    return *this;
  };

  Result(Result &&) = default;
  Result &operator=(Result &&) = default;

  bool ok() const noexcept { return (mStatus == Status::kOk); }

  Status status() { return mStatus; }

  void set(Status r) noexcept { mStatus = r; }

  std::string message() const noexcept { return mMsg.str(); }

  template <typename... Ts>
  void addMessage(Ts... args)
  {
    std::initializer_list<int>{(mMsg << args, 0)...};
  }

  void reset()
  {
    std::stringstream newMsg;
    std::swap(mMsg, newMsg);
    mStatus = Status::kOk;
  }

private:
  Status            mStatus = Status::kOk;
  std::stringstream mMsg;
};

class MessageList
{
public:
  template <typename... Args>
  void warn(Args &&... args)
  {
    add<Result::Status::kWarning>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void error(Args &&... args)
  {
    add<Result::Status::kError>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void info(Args &&... args)
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
  void add(Args &&... args)
  {
    messages.emplace_back(S, std::forward<Args>(args)...);
  }

  std::vector<Result> messages;
};
} // namespace client
} // namespace fluid

