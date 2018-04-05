#pragma once
#include <vector>



template <class T,class Coll>
T prod(Coll& c) {
  T acc = 1;
  for (auto ele : c)
    acc *= ele;
  return acc;
}

