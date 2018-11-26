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
Rcpp::XPtr<torch::Tensor> tensor_ (SEXP x, std::vector<int64_t> dim, bool clone = true) {

  switch (TYPEOF(x)) {
  case INTSXP:
    return tensor_impl_<INTSXP, at::kInt>(x, dim, clone);
  case REALSXP:
    return tensor_impl_<REALSXP, at::kDouble>(x, dim, clone);
  default:
    Rcpp::stop("not handled");
  }
};

// [[Rcpp::export]]
void print_tensor_ (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor ten = *x;
  Rcpp::Rcout << ten << std::endl;
};

template <int RTYPE, typename STDTYPE>
Rcpp::List as_array_tensor_impl_ (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor ten = *x;

  Rcpp::IntegerVector dimensions(ten.ndimension());
  for (int i = 0; i < ten.ndimension(); ++i) {
    dimensions[i] = ten.size(i);
  }

  ten = ten.contiguous();
  Rcpp::Vector<RTYPE> vec(ten.data<STDTYPE>(), ten.data<STDTYPE>() + ten.numel());
  vec = clone(vec);

  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = dimensions);
}

// [[Rcpp::export]]
Rcpp::List as_array_tensor_ (Rcpp::XPtr<torch::Tensor> x) {

  torch::Tensor ten = *x;

  switch (ten.dtype()) {
  case torch::kInt:
    return as_array_tensor_impl_<INTSXP, int32_t>(x);
  case torch::kDouble:
    return as_array_tensor_impl_<REALSXP, double>(x);
  default:
    Rcpp::stop("not handled");
  }

};


