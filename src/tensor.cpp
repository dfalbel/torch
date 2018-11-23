#include "torch_types.h"

template <int RTYPE, at::ScalarType ATTYPE>
Rcpp::XPtr<torch::Tensor> tensor_impl (SEXP x, bool clone = true) {

  Rcpp::Vector<RTYPE> vec(x);
  std::vector<int64_t> dim = vec.attr("dim");

  auto tensor = torch::from_blob(vec.begin(), dim, ATTYPE);

  if (clone)
    tensor = tensor.clone();

  auto * ten = new torch::Tensor(tensor);
  auto ptr = Rcpp::XPtr<torch::Tensor>(ten);

  return ptr;
};


// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor (SEXP x) {

  switch (TYPEOF(x)) {
  case INTSXP:
    return tensor_impl<INTSXP, at::kInt>(x);
  case REALSXP:
    return tensor_impl<REALSXP, at::kDouble>(x);
  default:
    Rcpp::stop("not handled");
  }
};
