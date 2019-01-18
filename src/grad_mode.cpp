#include "torch_types.h"

// [[Rcpp::export]]
void set_grad_mode (bool enabled) {
  torch::autograd::GradMode::set_enabled(enabled);
}
