#include "torch_types.h"

// [[Rcpp::export]]
bool cuda_is_available_ () {
  return torch::cuda::is_available();
}
