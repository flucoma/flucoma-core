#pragma once

namespace fluid
{
namespace client
{
  constexpr auto  NoBufferError = "No buffer passed";
  constexpr auto  PointNotFoundError = "Point not found";
  constexpr auto  WrongPointSizeError = "Wrong Point Size";
  constexpr auto  WrongInitError = "Wrong number of initial points";
  constexpr auto  DuplicateError = "Label already in dataset";
  constexpr auto  SmallDatasetError = "Dataset is smaller than k";
  constexpr auto  EmptyDatasetError = "Dataset is empty";
  constexpr auto  WriteError = "Couldn't write file";
  constexpr auto  ReadError = "Couldn't read file";
}
}
