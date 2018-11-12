#pragma once

#include <torch/torch.h>
#include <Rcpp.h>

typedef Rcpp::XPtr<std::shared_ptr<torch::Tensor>> tensor_ptr;
