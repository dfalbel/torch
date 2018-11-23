#include "torch_types.h"

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_impl (SEXP x, bool clone = true) {

  Rcpp::Vector<REALSXP> vec(x);
  std::vector<int64_t> dim = vec.attr("dim");

  auto tensor = torch::from_blob(vec.begin(), dim, at::kDouble);

  if (clone)
    tensor = tensor.clone();

  auto * ten = new torch::Tensor(tensor);
  auto ptr = Rcpp::XPtr<torch::Tensor>(ten);

  return ptr;
}
