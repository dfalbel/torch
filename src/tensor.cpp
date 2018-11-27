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
void tensor_print_ (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor ten = *x;
  Rcpp::Rcout << ten << std::endl;
};

template <int RTYPE, typename STDTYPE>
Rcpp::List as_array_tensor_impl_ (Rcpp::XPtr<torch::Tensor> x) {

  Rcpp::IntegerVector dimensions(x->ndimension());
  for (int i = 0; i < x->ndimension(); ++i) {
    dimensions[i] = x->size(i);
  }

  auto ten = x->contiguous();
  Rcpp::Vector<RTYPE> vec(ten.data<STDTYPE>(), ten.data<STDTYPE>() + ten.numel());

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

Rcpp::XPtr<torch::Tensor> make_tensor_ptr (torch::Tensor x) {
  auto * out = new torch::Tensor(x);
  return Rcpp::XPtr<torch::Tensor>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_abs_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->abs());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_acos_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->acos());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_add_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> y) {
  return make_tensor_ptr(x->add(*y));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addbmm_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> batch1,
                                    Rcpp::XPtr<torch::Tensor> batch2, double beta, double alpha) {
  return make_tensor_ptr(x->addbmm(*batch1, *batch2, beta, alpha));
}

// [[Rcpp::export]]
std::string tensor_to_string_ (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor ten = *x;
  return ten.toString();
}

