#include "torch_types.h"

template <int RTYPE, at::ScalarType ATTYPE>
Rcpp::XPtr<torch::Tensor> tensor_impl_ (SEXP x, std::vector<int64_t> dim, bool clone = true) {

  auto rtype = RTYPE;
  auto attype = ATTYPE;

  // since R logical vectors have 8B we need to treat them as integer vectors
  // and then cast to bit tensor.
  if (RTYPE == LGLSXP)
    attype = torch::kInt32;

  Rcpp::Vector<RTYPE> vec(x);

  auto tensor = torch::from_blob(vec.begin(), dim, attype);

  if (clone)
    tensor = tensor.clone();

  if (RTYPE == LGLSXP)
    tensor = tensor.to(torch::kByte);

  auto * ten = new torch::Tensor(tensor);
  auto ptr = Rcpp::XPtr<torch::Tensor>(ten);

  return ptr;
};


// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ (SEXP x, std::vector<int64_t> dim, bool clone = true) {

  switch (TYPEOF(x)) {
  case INTSXP:
    return tensor_impl_<INTSXP, torch::kInt>(x, dim, clone);
  case REALSXP:
    return tensor_impl_<REALSXP, torch::kDouble>(x, dim, clone);
  case LGLSXP:
    return tensor_impl_<LGLSXP, torch::kByte>(x, dim, clone);
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

  if (ten.dtype() == torch::kLong)
    ten = ten.to(torch::kInt);

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
  case torch::kByte:
    // TODO:
    // not sure why this works :(
    return as_array_tensor_impl_<LGLSXP, std::uint8_t>(x);
  case torch::kLong:
    // TODO: deal better with kLongs
    // Klong is casted to kInt inside impl
    return as_array_tensor_impl_<INTSXP, int32_t>(x);
  default:
    Rcpp::stop("not handled");
  };

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
Rcpp::XPtr<torch::Tensor> tensor_addcdiv_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> tensor1,
                                           Rcpp::XPtr<torch::Tensor> tensor2, double value
) {
  return make_tensor_ptr(x->addcdiv(*tensor1, *tensor2, value));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addcmul_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> tensor1,
                                           Rcpp::XPtr<torch::Tensor> tensor2, double value
) {
  return make_tensor_ptr(x->addcmul(*tensor1, *tensor2, value));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addmm_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> mat1,
                                         Rcpp::XPtr<torch::Tensor> mat2, double beta, double alpha
) {
  return make_tensor_ptr(x->addmm(*mat1, *mat2, beta, alpha));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addmv_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> mat,
                                         Rcpp::XPtr<torch::Tensor> vec, double beta, double alpha) {
  return make_tensor_ptr(x->addmv(*mat, *vec, beta, alpha));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addr_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> vec1,
                                        Rcpp::XPtr<torch::Tensor> vec2, double beta, double alpha) {
  return make_tensor_ptr(x->addr(*vec1, *vec2, beta, alpha));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_all_ (Rcpp::XPtr<torch::Tensor> x, std::int64_t dim, bool keepdim) {
  if (dim < 0)
    return make_tensor_ptr(x->all());
  else
    return make_tensor_ptr(x->all(dim, keepdim));
}

// [[Rcpp::export]]
bool tensor_allclose_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other,
                       double rtol, double atol, bool equal_nan) {
  return x->allclose(*other, rtol, atol, equal_nan);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_any_ (Rcpp::XPtr<torch::Tensor> x, std::int64_t dim, bool keepdim) {
  if (dim < 0)
    return make_tensor_ptr(x->any());
  else
    return make_tensor_ptr(x->any(dim, keepdim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_argmax_ (Rcpp::XPtr<torch::Tensor> x, std::int64_t dim, bool keepdim) {
  if (dim < 0)
    return make_tensor_ptr(x->argmax());
  else
    return make_tensor_ptr(x->argmax(dim, keepdim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_argmin_ (Rcpp::XPtr<torch::Tensor> x, std::int64_t dim, bool keepdim) {
  if (dim < 0)
    return make_tensor_ptr(x->argmin());
  else
    return make_tensor_ptr(x->argmin(dim, keepdim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_as_strided_ (Rcpp::XPtr<torch::Tensor> x,
                                              Rcpp::IntegerVector size,
                                              Rcpp::IntegerVector stride,
                                              Rcpp::Nullable<Rcpp::IntegerVector> storage_offset
) {

  torch::IntList size2 = Rcpp::as<std::vector<int64_t>>(size);
  torch::IntList stride2 = Rcpp::as<std::vector<int64_t>>(stride);

  if (storage_offset.isNull())
    return make_tensor_ptr(x->as_strided(size2, stride2));
  else {
    int64_t storage_offset2 = Rcpp::as<int64_t>(storage_offset);
    return make_tensor_ptr(x->as_strided(size2, stride2, storage_offset2));
  }
}

// [[Rcpp::export]]
std::string tensor_to_string_ (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor ten = *x;
  return ten.toString();
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_asin_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->asin());
}
