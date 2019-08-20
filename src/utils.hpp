#include "torch_types.h"

Rcpp::XPtr<torch::Tensor> make_tensor_ptr (torch::Tensor x) {
  auto * out = new torch::Tensor(x);
  return Rcpp::XPtr<torch::Tensor>(out);
}

std::vector<torch::Tensor> tensor_list_from_r_ (Rcpp::List x) {
  std::vector<torch::Tensor> out;
  for (int i = 0; i < x.size(); i++) {
    auto tmp = Rcpp::as<Rcpp::XPtr<torch::Tensor>>(x[i]);
    out.push_back(*tmp);
  }
  return out;
}
