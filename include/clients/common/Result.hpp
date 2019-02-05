#pragma once

namespace fluid {
namespace client {

class Result {
public:
  enum class Status {kOk, kWarning, kError};
  
  Result(Status status, std::string msg):mStatus(status),mMsg(msg){}

  template <typename...Ts>
  Result(Status status, Ts...msgChunks):mStatus(status)
  {
   std::initializer_list<int>{(addMessage(msgChunks),0)...};
  }


  Result() = default;
  Result(Result& x):mStatus(x.mStatus),mMsg(x.mMsg.str())
  {}
  
  Result operator=(Result& r)
  {
    return {r}; 
  };


  Result(Result&&) = default;
  Result& operator=(Result&&)=default;

  
  bool ok() const noexcept { return (mStatus == Status::kOk); }
  
  Status status() { return mStatus; }
  
  void set(Status r) noexcept { mStatus = r; }
  std::string message() const noexcept { return mMsg.str(); }
  template<typename T>
  void addMessage(T str)
  {
    mMsg << str;
  }
  
  void reset()
  {
    mMsg.clear();
    mStatus = Status::kOk;
  }
  
private:
  Status mStatus = Status::kOk;
  std::stringstream mMsg;
};

}
}
