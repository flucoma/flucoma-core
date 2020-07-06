#pragma once

#include <clients/common/Result.hpp>

namespace fluid {
namespace client {
  static const std::string NoBuffer{"No buffer passed"};
  static const std::string InvalidBuffer{"Invalid buffer"};
  static const std::string EmptyBuffer{"Empty buffer"};
  static const std::string PointNotFound{"Point not found"};
  static const std::string WrongPointSize{"Wrong Point Size"};
  static const std::string WrongPointNumber{"Wrong number of points"};
  static const std::string WrongNumInitial{"Wrong number of initial points"};
  static const std::string DuplicateLabel{"Label already in dataset"};
  static const std::string SmallDataSet{"DataSet is smaller than k"};
  static const std::string SmallK{"k is too small"};
  static const std::string EmptyDataSet{"DataSet is empty"};
  static const std::string EmptyLabelSet{"LabelSet is empty"};
  static const std::string NoDataSet{"DataSet does not exist"};
  static const std::string NoLabelSet{"LabelSet does not exist"};
  static const std::string NoDataFitted{"No data fitted"};
  static const std::string NotEnoughData{"Not enough data"};
  static const std::string EmptyLabel{"Empty label"};
  static const std::string EmptyId{"Empty id"};
  static const std::string BufferAlloc{"Can't allocate buffer"};
  static const std::string FileRead{"Couldn't read file"};
  static const std::string FileWrite{"Couldn't write file"};
  static const std::string NotImplemented{"Not implemented"};
  static const std::string SizesDontMatch{"Sizes do not match"};

  template <typename T>
  MessageResult<T> Error(std::string msg){
    return MessageResult<T>{Result::Status::kError,msg};
  };

  MessageResult<void> Error(std::string msg){
    return MessageResult<void>{Result::Status::kError,msg};
  };
  MessageResult<void> OK(){
    return MessageResult<void>{Result::Status::kOk};
  }
} // namespace client
} // namespace fluid
