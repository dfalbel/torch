#include "torch_types.h"

template <int RTYPE, at::ScalarType ATTYPE>
Rcpp::XPtr<torch::Tensor> tensor_impl_ (SEXP x, std::vector<int64_t> dim, bool clone = true) {

  Rcpp::Vector<RTYPE> vec(x);

  auto tensor = torch::from_blob(vec.begin(), dim, ATTYPE);

  if (clone)
    tensor = tensor.clone();

  auto * ten = new torch::Tensor(tensor);
  auto ptr = Rcpp::XPtr<torch::Tensor>(ten);

  return ptr;
};


// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ (SEXP x, std::vector<int64_t> dim) {

  switch (TYPEOF(x)) {
  case INTSXP:
    return tensor_impl_<INTSXP, at::kInt>(x, dim);
  case REALSXP:
    return tensor_impl_<REALSXP, at::kDouble>(x, dim);
  default:
    Rcpp::stop("not handled");
  }
};


