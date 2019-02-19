#include "torch_types.h"
#include "scalar.hpp"
#include "device.hpp"

// Some utils ------------------------------------------------------------------

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

// Tensor from R code ----------------------------------------------------------

std::vector<int64_t> reverse_int_seq (int n) {
  std::vector<int64_t> l(n);
  std::iota(l.begin(), l.end(), 0);
  std::reverse(l.begin(), l.end());
  return l;
};

torch::Layout layout_from_string (std::string layout) {
  if (layout == "strided") {
    return torch::Layout::Strided;
  } else if (layout == "sparse") {
    return torch::Layout::Sparse;
  } else {
    Rcpp::stop("Layout type not implemented.");
  }
}

torch::TensorOptions tensor_options_ (Rcpp::Nullable<std::string> dtype,
                                      Rcpp::Nullable<std::string> layout,
                                      Rcpp::Nullable<std::string> device,
                                      Rcpp::Nullable<bool> requires_grad) {

  auto options = torch::TensorOptions();

  if (dtype.isNotNull()) {
    options = options.dtype(scalar_type_from_string(Rcpp::as<std::string>(dtype)));
  }

  if (layout.isNotNull()) {
    options = options.layout(layout_from_string(Rcpp::as<std::string>(layout)));
  }

  if (device.isNotNull()) {
    options = options.device(device_from_string(Rcpp::as<std::string>(device)));
  }

  if (requires_grad.isNotNull()) {
    options = options.requires_grad(Rcpp::as<bool>(requires_grad));
  }

  return options;
}

template <int RTYPE, torch::ScalarType SCALARTYPE>
torch::Tensor tensor_from_r_impl_ (const SEXP x, const std::vector<int64_t> dim) {

  Rcpp::Vector<RTYPE> vec(x);

  auto tensor = torch::from_blob(vec.begin(), dim, SCALARTYPE);

  if (dim.size() == 1) {
    // if we have a 1-dim vector contigous doesn't trigger a copy, and
    // would be unexpected.
    tensor = tensor.clone();
  }

  tensor = tensor
    .permute(reverse_int_seq(dim.size()))
    .contiguous(); // triggers a copy!

  return tensor;
};

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_from_r_ (SEXP x, std::vector<int64_t> dim,
                                          Rcpp::Nullable<std::string> dtype,
                                          Rcpp::Nullable<std::string> device,
                                          bool requires_grad = false
) {

  torch::Tensor tensor;

  if (TYPEOF(x) == INTSXP) {
    tensor = tensor_from_r_impl_<INTSXP, torch::kInt>(x, dim);
  } else if (TYPEOF(x) == REALSXP) {
    tensor = tensor_from_r_impl_<REALSXP, torch::kDouble>(x, dim);
  } else if (TYPEOF(x) == LGLSXP) {
    tensor = tensor_from_r_impl_<LGLSXP, torch::kInt32>(x, dim);
  } else {
    Rcpp::stop("R type not handled");
  };

  torch::TensorOptions options = tensor_options_(dtype, R_NilValue, device, R_NilValue);

  if (dtype.isNull()) {
    if (TYPEOF(x) == REALSXP) {
      options = options.dtype(torch::kFloat);
    } else if (TYPEOF(x) == LGLSXP) {
      options = options.dtype(torch::kByte);
    }
  }

  tensor = tensor.to(options);
  tensor = tensor.set_requires_grad(requires_grad);

  return make_tensor_ptr(tensor);
};

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ (Rcpp::XPtr<torch::Tensor> x,
                                   Rcpp::Nullable<std::string> dtype,
                                   Rcpp::Nullable<std::string> device,
                                   bool requires_grad) {
  torch::Tensor tensor = x->clone();
  tensor = tensor.to(tensor_options_(dtype, R_NilValue, device, R_NilValue));
  tensor = tensor.set_requires_grad(requires_grad);
  return make_tensor_ptr(tensor);
}

// Tensor to R code ------------------------------------------------------------

template <int RTYPE, typename STDTYPE>
Rcpp::List as_array_tensor_impl_ (torch::Tensor x) {

  Rcpp::IntegerVector dimensions(x.ndimension());
  for (int i = 0; i < x.ndimension(); ++i) {
    dimensions[i] = x.size(i);
  }

  auto ten = x.contiguous();

  Rcpp::Vector<RTYPE> vec(ten.data<STDTYPE>(), ten.data<STDTYPE>() + ten.numel());

  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = dimensions);
}

// [[Rcpp::export]]
Rcpp::List as_array_tensor_ (Rcpp::XPtr<torch::Tensor> x) {

  torch::Tensor ten = *x;

  if (ten.dtype() == torch::kInt) {
    return as_array_tensor_impl_<INTSXP, int32_t>(ten);
  } else if (ten.dtype() == torch::kDouble) {
    return as_array_tensor_impl_<REALSXP, double>(ten);
  } else if (ten.dtype() == torch::kByte) {
    return as_array_tensor_impl_<LGLSXP, std::uint8_t>(ten);
  } else if (ten.dtype() == torch::kLong) {
    return as_array_tensor_impl_<INTSXP, int32_t>(ten.to(torch::kInt));
  } else if (ten.dtype() == torch::kFloat) {
    return as_array_tensor_impl_<REALSXP, double>(ten.to(torch::kDouble));
  }

  Rcpp::stop("dtype not handled");
};

// Tensor Creation -------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_randn_ (std::vector<std::int64_t> size,
                                        Rcpp::Nullable<std::string> dtype,
                                        Rcpp::Nullable<std::string> layout,
                                        Rcpp::Nullable<std::string> device,
                                        Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::randn(size, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_arange_ (SEXP start,
                                         SEXP end,
                                         SEXP step,
                                         Rcpp::Nullable<std::string> dtype,
                                         Rcpp::Nullable<std::string> layout,
                                         Rcpp::Nullable<std::string> device,
                                         Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::arange(scalar_from_r_(start), scalar_from_r_(end), scalar_from_r_(step), tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_empty_ (std::vector<std::int64_t> size,
                                        Rcpp::Nullable<std::string> dtype,
                                        Rcpp::Nullable<std::string> layout,
                                        Rcpp::Nullable<std::string> device,
                                        Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::empty(size, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_eye_ (std::int64_t n,
                                      std::int64_t m,
                                      Rcpp::Nullable<std::string> dtype,
                                      Rcpp::Nullable<std::string> layout,
                                      Rcpp::Nullable<std::string> device,
                                      Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::eye(n, m, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_full_ (std::vector<std::int64_t> size,
                                       SEXP fill_value,
                                       Rcpp::Nullable<std::string> dtype,
                                       Rcpp::Nullable<std::string> layout,
                                       Rcpp::Nullable<std::string> device,
                                       Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::full(size, scalar_from_r_(fill_value), tensor_options_(dtype, layout, device, requires_grad)));
}


// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_linspace_ (SEXP start,
                                           SEXP end,
                                           std::int64_t steps,
                                           Rcpp::Nullable<std::string> dtype,
                                           Rcpp::Nullable<std::string> layout,
                                           Rcpp::Nullable<std::string> device,
                                           Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::linspace(scalar_from_r_(start), scalar_from_r_(end), steps, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_logspace_ (SEXP start,
                                           SEXP end,
                                           std::int64_t steps,
                                           Rcpp::Nullable<std::string> dtype,
                                           Rcpp::Nullable<std::string> layout,
                                           Rcpp::Nullable<std::string> device,
                                           Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::logspace(scalar_from_r_(start), scalar_from_r_(end), steps, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_ones_ (std::vector<std::int64_t> size,
                                       Rcpp::Nullable<std::string> dtype,
                                       Rcpp::Nullable<std::string> layout,
                                       Rcpp::Nullable<std::string> device,
                                       Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::ones(size, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_rand_ (std::vector<std::int64_t> size,
                                       Rcpp::Nullable<std::string> dtype,
                                       Rcpp::Nullable<std::string> layout,
                                       Rcpp::Nullable<std::string> device,
                                       Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::rand(size, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_randint_ (std::int64_t low,
                                          std::int64_t high,
                                          std::vector<std::int64_t> size,
                                          Rcpp::Nullable<std::string> dtype,
                                          Rcpp::Nullable<std::string> layout,
                                          Rcpp::Nullable<std::string> device,
                                          Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::randint(low, high, size, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_randperm_ (std::int64_t n,
                                           Rcpp::Nullable<std::string> dtype,
                                           Rcpp::Nullable<std::string> layout,
                                           Rcpp::Nullable<std::string> device,
                                           Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::randperm(n, tensor_options_(dtype, layout, device, requires_grad)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> torch_zeros_ (std::vector<std::int64_t> size,
                                        Rcpp::Nullable<std::string> dtype,
                                        Rcpp::Nullable<std::string> layout,
                                        Rcpp::Nullable<std::string> device,
                                        Rcpp::Nullable<bool> requires_grad
) {
  return make_tensor_ptr(torch::zeros(size, tensor_options_(dtype, layout, device, requires_grad)));
}

// Tensor Methods --------------------------------------------------------------

// [[Rcpp::export]]
void tensor_print_ (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor ten = *x;
  Rcpp::Rcout << ten << std::endl;
};

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_abs_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->abs());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_abs__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->abs_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_acos_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->acos());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_acos__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->acos_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_add_tensor_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> y) {
  return make_tensor_ptr(x->add(*y));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_add_scalar_ (Rcpp::XPtr<torch::Tensor> x, SEXP y) {
  return make_tensor_ptr(x->add(scalar_from_r_(y)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_add_tensor__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> y) {
  return make_tensor_ptr(x->add_(*y));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_add_scalar__ (Rcpp::XPtr<torch::Tensor> x, SEXP y) {
  return make_tensor_ptr(x->add_(scalar_from_r_(y)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addbmm_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> batch1,
                                          Rcpp::XPtr<torch::Tensor> batch2, SEXP beta, SEXP alpha) {
  return make_tensor_ptr(x->addbmm(*batch1, *batch2, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addbmm__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> batch1,
                                           Rcpp::XPtr<torch::Tensor> batch2, SEXP beta, SEXP alpha) {
  return make_tensor_ptr(x->addbmm_(*batch1, *batch2, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addcdiv_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> tensor1,
                                           Rcpp::XPtr<torch::Tensor> tensor2, SEXP value) {
  return make_tensor_ptr(x->addcdiv(*tensor1, *tensor2, scalar_from_r_(value)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addcdiv__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> tensor1,
                                            Rcpp::XPtr<torch::Tensor> tensor2, SEXP value) {
  return make_tensor_ptr(x->addcdiv_(*tensor1, *tensor2, scalar_from_r_(value)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addcmul_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> tensor1,
                                           Rcpp::XPtr<torch::Tensor> tensor2, SEXP value
) {
  return make_tensor_ptr(x->addcmul(*tensor1, *tensor2, scalar_from_r_(value)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addcmul__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> tensor1,
                                            Rcpp::XPtr<torch::Tensor> tensor2, SEXP value
) {
  return make_tensor_ptr(x->addcmul_(*tensor1, *tensor2, scalar_from_r_(value)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addmm_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> mat1,
                                         Rcpp::XPtr<torch::Tensor> mat2, SEXP beta, SEXP alpha
) {
  return make_tensor_ptr(x->addmm(*mat1, *mat2, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addmm__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> mat1,
                                          Rcpp::XPtr<torch::Tensor> mat2, SEXP beta, SEXP alpha
) {
  return make_tensor_ptr(x->addmm_(*mat1, *mat2, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addmv_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> mat,
                                         Rcpp::XPtr<torch::Tensor> vec, SEXP beta, SEXP alpha) {
  return make_tensor_ptr(x->addmv(*mat, *vec, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addmv__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> mat,
                                          Rcpp::XPtr<torch::Tensor> vec, SEXP beta, SEXP alpha) {
  return make_tensor_ptr(x->addmv_(*mat, *vec, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addr_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> vec1,
                                        Rcpp::XPtr<torch::Tensor> vec2, SEXP beta, SEXP alpha) {
  return make_tensor_ptr(x->addr(*vec1, *vec2, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_addr__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> vec1,
                                         Rcpp::XPtr<torch::Tensor> vec2, SEXP beta, SEXP alpha) {
  return make_tensor_ptr(x->addr_(*vec1, *vec2, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_all_ (Rcpp::XPtr<torch::Tensor> x,
                                       Rcpp::Nullable<std::int64_t> dim,
                                       bool keepdim) {

  if (dim.isNull())
    return make_tensor_ptr(x->all());
  else {
    return make_tensor_ptr(x->all(Rcpp::as<int64_t>(dim), keepdim));
  }
}

// [[Rcpp::export]]
bool tensor_allclose_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other,
                       double rtol, double atol, bool equal_nan) {
  return x->allclose(*other, rtol, atol, equal_nan);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_any_ (Rcpp::XPtr<torch::Tensor> x,
                                       Rcpp::Nullable<int64_t> dim,
                                       bool keepdim) {
  if (dim.isNull())
    return make_tensor_ptr(x->any());
  else {
    return make_tensor_ptr(x->any(Rcpp::as<std::int64_t>(dim), keepdim));
  }
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_argmax_ (Rcpp::XPtr<torch::Tensor> x,
                                          Rcpp::Nullable<std::int64_t> dim,
                                          bool keepdim) {
  if (dim.isNull())
    return make_tensor_ptr(x->argmax());
  else {
    return make_tensor_ptr(x->argmax(Rcpp::as<std::int64_t>(dim), keepdim));
  }
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_argmin_ (Rcpp::XPtr<torch::Tensor> x,
                                          Rcpp::Nullable<std::int64_t> dim,
                                          bool keepdim) {
  if (dim.isNull())
    return make_tensor_ptr(x->argmin());
  else {
    return make_tensor_ptr(x->argmin(Rcpp::as<std::int64_t>(dim), keepdim));
  }
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_asin_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->asin());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_asin__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->asin_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_atan_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->atan());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_atan__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->atan_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_atan2_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->atan2(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_atan2__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->atan2_(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_as_strided_ (Rcpp::XPtr<torch::Tensor> x,
                                              std::vector<int64_t> size,
                                              std::vector<int64_t> stride,
                                              Rcpp::Nullable<int64_t> storage_offset
) {
  if (storage_offset.isNull())
    return make_tensor_ptr(x->as_strided(size, stride));
  else {
    return make_tensor_ptr(x->as_strided(size, stride, Rcpp::as<int64_t>(storage_offset)));
  }
}

// [[Rcpp::export]]
std::string tensor_to_string_ (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor ten = *x;
  return ten.toString();
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sin_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sin());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sin__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sin_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sinh_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sinh());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sinh__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sinh_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_tan_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->tan());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_tan__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->tan_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_tanh_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->tanh());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_tanh__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->tanh_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_trunc_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->trunc());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_trunc__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->trunc_());
}

// [[Rcpp::export]]
void tensor_backward_ (Rcpp::XPtr<torch::Tensor> x,
                       Rcpp::Nullable<Rcpp::XPtr<torch::Tensor>> gradient,
                       bool keep_graph, bool create_graph) {
  if(gradient.isNull()) {
    x->backward(at::nullopt, keep_graph, create_graph);
  } else {
    Rcpp::XPtr<torch::Tensor> gradient2(gradient.get());
    x->backward(*gradient2, keep_graph, create_graph);
  }

}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_baddbmm_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> batch1,
                                           Rcpp::XPtr<torch::Tensor> batch2, SEXP beta, SEXP alpha) {
  return make_tensor_ptr(x->baddbmm(*batch1, *batch2, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_baddbmm__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> batch1,
                                            Rcpp::XPtr<torch::Tensor> batch2, SEXP beta, SEXP alpha) {
  return make_tensor_ptr(x->baddbmm_(*batch1, *batch2, scalar_from_r_(beta), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_bernoulli_ (Rcpp::XPtr<torch::Tensor> x,
                                             Rcpp::Nullable<double> p) {
  if (p.isNull()) {
    return make_tensor_ptr(x->bernoulli());
  } else {
    return make_tensor_ptr(x->bernoulli(Rcpp::as<double>(p)));
  }
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_bernoulli_double__ (Rcpp::XPtr<torch::Tensor> x, double p = 0.5) {
  return make_tensor_ptr(x->bernoulli_(p));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_bernoulli_tensor__ (Rcpp::XPtr<torch::Tensor> x,
                                                     Rcpp::XPtr<torch::Tensor> p) {
  return make_tensor_ptr(x->bernoulli_(*p));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_bincount_ (Rcpp::XPtr<torch::Tensor> x,
                                            Rcpp::Nullable<Rcpp::XPtr<torch::Tensor>> weights,
                                            std::int64_t minlength) {

  if (weights.isNull()) {
    return make_tensor_ptr(x->bincount({}, minlength));
  } else {
    return make_tensor_ptr(x->bincount(*(Rcpp::XPtr<torch::Tensor>(weights.get())), minlength));
  }
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_bmm_ (Rcpp::XPtr<torch::Tensor> x,
                                       Rcpp::XPtr<torch::Tensor> mat2) {
  return make_tensor_ptr(x->bmm(*mat2));
}

// [[Rcpp::export]]
Rcpp::List tensor_btrifact_ (Rcpp::XPtr<torch::Tensor> x, bool pivot) {
  auto out = x->btrifact(pivot);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
Rcpp::List tensor_btrifact_with_info_ (Rcpp::XPtr<torch::Tensor> x, bool pivot) {
  auto out = x->btrifact_with_info(pivot);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)), make_tensor_ptr(std::get<2>(out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_byte_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->to(torch::kByte));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_btrisolve_ (Rcpp::XPtr<torch::Tensor> x,
                                             Rcpp::XPtr<torch::Tensor> LU_data,
                                             Rcpp::XPtr<torch::Tensor> LU_pivots) {
  return make_tensor_ptr(x->btrisolve(*LU_data, *LU_pivots));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cauchy__ (Rcpp::XPtr<torch::Tensor> x, double median, double sigma) {
  return make_tensor_ptr(x->cauchy_(median, sigma));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ceil_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->ceil());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ceil__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->ceil_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_char_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->to(torch::kChar));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cholesky_ (Rcpp::XPtr<torch::Tensor> x, bool upper = false) {
  return make_tensor_ptr(x->cholesky(upper));
}

// [[Rcpp::export]]
Rcpp::List tensor_chunk_ (Rcpp::XPtr<torch::Tensor> x, int64_t chunks, int64_t dim) {

  auto chunks_vector = x->chunk(chunks, dim);
  Rcpp::List out;

  for (int i = 0; i < chunks_vector.size(); ++i) {
    out.push_back(make_tensor_ptr(chunks_vector[i]));
  }

  return out;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_clamp_ (Rcpp::XPtr<torch::Tensor> x,
                                         SEXP min,
                                         SEXP max) {
  return make_tensor_ptr(x->clamp(scalar_from_r_(min), scalar_from_r_(max)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_clamp__ (Rcpp::XPtr<torch::Tensor> x,
                                          SEXP min,
                                          SEXP max) {
  return make_tensor_ptr(x->clamp_(scalar_from_r_(min), scalar_from_r_(max)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_clamp_max_ (Rcpp::XPtr<torch::Tensor> x, SEXP max) {
  return make_tensor_ptr(x->clamp_max(scalar_from_r_(max)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_clamp_max__ (Rcpp::XPtr<torch::Tensor> x, SEXP max) {
  return make_tensor_ptr(x->clamp_max_(scalar_from_r_(max)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_clamp_min_ (Rcpp::XPtr<torch::Tensor> x, SEXP min) {
  return make_tensor_ptr(x->clamp_min(scalar_from_r_(min)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_clamp_min__ (Rcpp::XPtr<torch::Tensor> x, SEXP min) {
  return make_tensor_ptr(x->clamp_min_(scalar_from_r_(min)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_clone_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->clone());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_contiguous_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->contiguous());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_copy__ (Rcpp::XPtr<torch::Tensor> x,
                                         Rcpp::XPtr<torch::Tensor> src,
                                         bool non_blocking = false) {
  return make_tensor_ptr(x->copy_(*src, non_blocking));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cos_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->cos());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cos__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->cos_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cosh_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->cosh());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cosh__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->cosh_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cpu_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->cpu());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cross_ (Rcpp::XPtr<torch::Tensor> x,
                                         Rcpp::XPtr<torch::Tensor> other,
                                         std::int64_t dim = -1) {
  return make_tensor_ptr(x->cross(*other, dim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cuda_ (Rcpp::XPtr<torch::Tensor> x) {
  if (torch::cuda::is_available()) {
    return make_tensor_ptr(x->cuda());
  }
  Rcpp::stop("CUDA is not available.");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cumprod_ (Rcpp::XPtr<torch::Tensor> x, std::int64_t dim) {
  // TODO allow dtype argument.
  return make_tensor_ptr(x->cumprod(dim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_cumsum_ (Rcpp::XPtr<torch::Tensor> x, std::int64_t dim) {
  // TODO allow dtype argument.
  return make_tensor_ptr(x->cumsum(dim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_data_ (Rcpp::XPtr<torch::Tensor> x) {
  auto out = torch::from_blob(x->data_ptr(), x->sizes(), x->type());
  return make_tensor_ptr(out);
}

// [[Rcpp::export]]
std::string tensor_data_ptr_ (Rcpp::XPtr<torch::Tensor> x) {
  std::ostringstream s;
  s << (x->data_ptr());
  return s.str();
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_det_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->det());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_detach_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->detach());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_detach__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->detach_());
}

// [[Rcpp::export]]
std::string tensor_device_ (Rcpp::XPtr<torch::Tensor> x) {
  return device_to_string(x->device());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_diag_ (Rcpp::XPtr<torch::Tensor> x,
                                        std::int64_t diagonal = 0) {
  return make_tensor_ptr(x->diag(diagonal));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_diag_embed_ (Rcpp::XPtr<torch::Tensor> x,
                                              std::int64_t offset = 0,
                                              std::int64_t dim1 = -2,
                                              std::int64_t dim2 = -1) {
  return make_tensor_ptr(x->diag_embed(offset, dim1, dim2));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_diagflat_ (Rcpp::XPtr<torch::Tensor> x,
                                            std::int64_t offset = 0) {
  return make_tensor_ptr(x->diagflat(offset));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_diagonal_ (Rcpp::XPtr<torch::Tensor> x,
                                            std::int64_t offset = 0,
                                            std::int64_t dim1 = 0,
                                            std::int64_t dim2 = 1) {
  return make_tensor_ptr(x->diagonal(offset, dim1, dim2));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_digamma_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->digamma());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_digamma__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->digamma_());
}

// [[Rcpp::export]]
std::int64_t tensor_dim_ (Rcpp::XPtr<torch::Tensor> x) {
  return x->dim();
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_dist_ (Rcpp::XPtr<torch::Tensor> x,
                                        Rcpp::XPtr<torch::Tensor> other,
                                        SEXP p) {
  return make_tensor_ptr(x->dist(*other, scalar_from_r_(p)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_div_tensor_ (Rcpp::XPtr<torch::Tensor> x,
                                              Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->div(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_div_scalar_ (Rcpp::XPtr<torch::Tensor> x,
                                              SEXP other) {
  return make_tensor_ptr(x->div(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_div_tensor__ (Rcpp::XPtr<torch::Tensor> x,
                                               Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->div_(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_div_scalar__ (Rcpp::XPtr<torch::Tensor> x,
                                               SEXP other) {
  return make_tensor_ptr(x->div_(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_dot_ (Rcpp::XPtr<torch::Tensor> x,
                                       Rcpp::XPtr<torch::Tensor> tensor) {
  return make_tensor_ptr(x->dot(*tensor));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_double_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->to(torch::kDouble));
}

// [[Rcpp::export]]
std::string tensor_dtype_ (Rcpp::XPtr<torch::Tensor> x) {
  return caffe_type_to_string(x->dtype());
}

// [[Rcpp::export]]
Rcpp::List tensor_eig_ (Rcpp::XPtr<torch::Tensor> x,
                        bool eigenvectors = false) {
  auto out = x->eig(eigenvectors);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
int tensor_element_size_ (Rcpp::XPtr<torch::Tensor> x) {
  return x->dtype().itemsize();
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_eq_scalar_ (Rcpp::XPtr<torch::Tensor> x, SEXP other) {
  return make_tensor_ptr(x->eq(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_eq_tensor_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->eq(*other));
}

// [[Rcpp::export]]
bool tensor_equal_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return x->equal(*other);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_erf_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->erf());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_erf__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->erf_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_erfc_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->erfc());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_erfc__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->erfc_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_erfinv_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->erfinv());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_erfinv__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->erfinv_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_exp_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->exp());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_exp__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->exp_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_log_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->log());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_log__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->log_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_log2_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->log2());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_log2__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->log2_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_log10_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->log10());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_log10__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->log10_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_log1p_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->log1p());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_log1p__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->log1p_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_expand_ (Rcpp::XPtr<torch::Tensor> x,
                                          std::vector<std::int64_t> size,
                                          bool implicit = false) {
  return make_tensor_ptr(x->expand(size, implicit));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_expand_as_ (Rcpp::XPtr<torch::Tensor> x,
                                             Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->expand_as(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_expm1_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->expm1());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_expm1__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->expm1_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_exponential__ (Rcpp::XPtr<torch::Tensor> x, double lambd = 1) {
  return make_tensor_ptr(x->exponential_(lambd));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_fill_scalar__ (Rcpp::XPtr<torch::Tensor> x, SEXP value) {
  return make_tensor_ptr(x->fill_(scalar_from_r_(value)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_fill_tensor__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> value) {
  return make_tensor_ptr(x->fill_(*value));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_flatten_ (Rcpp::XPtr<torch::Tensor> x, int64_t start_dim, int64_t end_dim = -1) {
  return make_tensor_ptr(x->flatten(start_dim, end_dim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_flip_ (Rcpp::XPtr<torch::Tensor> x, std::vector<int64_t> dims) {
  return make_tensor_ptr(x->flip(dims));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_float_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->to(torch::kFloat));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_floor_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->floor());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_floor__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->floor_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_fmod_scalar_ (Rcpp::XPtr<torch::Tensor> x, SEXP other) {
  return make_tensor_ptr(x->fmod(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_fmod_tensor_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->fmod(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_fmod_scalar__ (Rcpp::XPtr<torch::Tensor> x, SEXP other) {
  return make_tensor_ptr(x->fmod_(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_fmod_tensor__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->fmod_(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_frac_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->frac());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_frac__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->frac_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_lerp_ (Rcpp::XPtr<torch::Tensor> start,
                                        Rcpp::XPtr<torch::Tensor> end,
                                        SEXP weight
) {
  return make_tensor_ptr(start->lerp(*end, scalar_from_r_(weight)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_gather_ (Rcpp::XPtr<torch::Tensor> x, int64_t dim, Rcpp::XPtr<torch::Tensor> index) {
  return make_tensor_ptr(x->gather(dim, *index));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ge_tensor_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->ge(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ge_scalar_ (Rcpp::XPtr<torch::Tensor> x, SEXP other) {
  return make_tensor_ptr(x->ge(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ge_tensor__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->ge_(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ge_scalar__ (Rcpp::XPtr<torch::Tensor> x, SEXP other) {
  return make_tensor_ptr(x->ge_(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::List tensor_gels_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> A) {
  auto out = x->gels(*A);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_grad_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->grad());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_geometric__ (Rcpp::XPtr<torch::Tensor> x, double p) {
  return make_tensor_ptr(x->geometric_(p));
}

// [[Rcpp::export]]
Rcpp::List tensor_geqrf_ (Rcpp::XPtr<torch::Tensor> x) {
  auto out = x->geqrf();
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_ger_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> vec2) {
  return make_tensor_ptr(x->ger(*vec2));
}

// [[Rcpp::export]]
Rcpp::List tensor_gesv_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> A) {
  auto out = x->gesv(*A);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
std::int64_t tensor_get_device_ (Rcpp::XPtr<torch::Tensor> x) {
  if (torch::cuda::is_available())
    return x->get_device();

  Rcpp::stop("get_device is not implemented for tensors with CPU backend.");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_gt_tensor_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->gt(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_gt_scalar_ (Rcpp::XPtr<torch::Tensor> x, SEXP other) {
  return make_tensor_ptr(x->gt(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_gt_tensor__ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->gt_(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_gt_scalar__ (Rcpp::XPtr<torch::Tensor> x, SEXP other) {
  return make_tensor_ptr(x->gt_(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_half_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->to(torch::kHalf));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_histc_ (Rcpp::XPtr<torch::Tensor> x,
                                         std::int64_t bins = 100,
                                         SEXP min = 0, SEXP max = 0) {
  return make_tensor_ptr(x->histc(bins, scalar_from_r_(min), scalar_from_r_(max)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_index_add__ (Rcpp::XPtr<torch::Tensor> x,
                                              std::int64_t dim,
                                              Rcpp::XPtr<torch::Tensor> index,
                                              Rcpp::XPtr<torch::Tensor> source
) {
  return make_tensor_ptr(x->index_add_(dim, *index, *source));
}


// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_index_copy__ (Rcpp::XPtr<torch::Tensor> x,
                                              std::int64_t dim,
                                              Rcpp::XPtr<torch::Tensor> index,
                                              Rcpp::XPtr<torch::Tensor> source
) {
  return make_tensor_ptr(x->index_copy_(dim, *index, *source));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_index_fill__ (Rcpp::XPtr<torch::Tensor> x,
                                               std::int64_t dim,
                                               Rcpp::XPtr<torch::Tensor> index,
                                               SEXP value) {
  return make_tensor_ptr(x->index_fill_(dim, *index, scalar_from_r_(value)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_index_put__ (Rcpp::XPtr<torch::Tensor> x,
                                              Rcpp::List indices,
                                              Rcpp::XPtr<torch::Tensor> values,
                                              bool accumulate = false) {
  return make_tensor_ptr(x->index_put_(tensor_list_from_r_(indices), *values, accumulate));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_index_select_ (Rcpp::XPtr<torch::Tensor> x,
                                              std::int64_t dim,
                                              Rcpp::XPtr<torch::Tensor> index) {
  return make_tensor_ptr(x->index_select(dim, *index));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_int_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->to(torch::kInt));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_inverse_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->inverse());
}

// [[Rcpp::export]]
bool tensor_is_contiguous_ (Rcpp::XPtr<torch::Tensor> x) {
  return x->is_contiguous();
}

// [[Rcpp::export]]
bool tensor_is_cuda_ (Rcpp::XPtr<torch::Tensor> x) {
  return x->is_cuda();
}

// [[Rcpp::export]]
bool tensor_is_set_to_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> tensor) {
  return x->is_set_to(*tensor);
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_mean_ (Rcpp::XPtr<torch::Tensor> x,
                                        Rcpp::Nullable<std::vector<std::int64_t>> dim,
                                        bool keepdim
) {

  if (dim.isNull()) {
    return make_tensor_ptr(x->mean());
  } else if (dim.isNotNull()) {
    return make_tensor_ptr(x->mean(Rcpp::as<std::vector<std::int64_t>>(dim), keepdim));
  }

  Rcpp::stop("Not yet implemented");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_var_ (Rcpp::XPtr<torch::Tensor> x,
                                       bool unbiased,
                                       Rcpp::Nullable<std::int64_t> dim,
                                       bool keepdim = false
) {


  if (dim.isNull()) {
    return make_tensor_ptr(x->var(unbiased));
  } else if (dim.isNotNull()) {
    return make_tensor_ptr(x->var(Rcpp::as<std::int64_t>(dim), unbiased, keepdim));
  }

  Rcpp::stop("Not yet implemented");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_std_ (Rcpp::XPtr<torch::Tensor> x,
                                       bool unbiased,
                                       Rcpp::Nullable<std::int64_t> dim,
                                       bool keepdim
) {

  if (dim.isNull()) {
    return make_tensor_ptr(x->std(unbiased));
  } else if (dim.isNotNull()) {
    return make_tensor_ptr(x->std(Rcpp::as<std::int64_t>(dim), unbiased, keepdim));
  }

  Rcpp::stop("Not yet implemented");
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_min_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->min());
}

// [[Rcpp::export]]
Rcpp::List tensor_min_dim_ (Rcpp::XPtr<torch::Tensor> x,
                            std::int64_t dim,
                            bool keepdim) {
  auto out = x->min(dim, keepdim);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_min_tensor_ (Rcpp::XPtr<torch::Tensor> x,
                                              Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->min(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_max_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->max());
}

// [[Rcpp::export]]
Rcpp::List tensor_max_dim_ (Rcpp::XPtr<torch::Tensor> x,
                            std::int64_t dim,
                            bool keepdim) {
  auto out = x->max(dim, keepdim);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_max_tensor_ (Rcpp::XPtr<torch::Tensor> x,
                                              Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->max(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_prod_ (Rcpp::XPtr<torch::Tensor> x,
                                        Rcpp::Nullable<std::int64_t> dim,
                                        Rcpp::Nullable<bool> keepdim,
                                        Rcpp::Nullable<std::string> dtype) {

  if(dim.isNull() & keepdim.isNull() & dtype.isNull()) { //2
    return make_tensor_ptr(x->prod());
  } else if(dim.isNull() & keepdim.isNull() & dtype.isNotNull()) { //1
    return make_tensor_ptr(x->prod(scalar_type_from_string(Rcpp::as<std::string>(dtype))));
  } else if(dim.isNotNull() & keepdim.isNotNull() & dtype.isNotNull()) { //5 ou //3
    return make_tensor_ptr(x->prod(Rcpp::as<std::int64_t>(dim), Rcpp::as<bool>(keepdim), scalar_type_from_string(Rcpp::as<std::string>(dtype))));
  } else if(dim.isNotNull() & keepdim.isNotNull() & dtype.isNull()) { //4
    return make_tensor_ptr(x->prod(Rcpp::as<std::int64_t>(dim), Rcpp::as<bool>(keepdim)));
  }

  Rcpp::stop("Not yet implemented");
}

// [[Rcpp::export]]
Rcpp::List tensor_median_dim_ (Rcpp::XPtr<torch::Tensor> x,
                               std::int64_t dim,
                               bool keepdim) {
  auto out = x->median(dim, keepdim);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_median_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->median());
}

// [[Rcpp::export]]
Rcpp::List tensor_mode_ (Rcpp::XPtr<torch::Tensor> x,
                         std::int64_t dim,
                         bool keepdim) {
  auto out = x->mode(dim, keepdim);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_logsumexp_ (Rcpp::XPtr<torch::Tensor> x,
                                             std::int64_t dim,
                                             bool keepdim) {
  return make_tensor_ptr(x->logsumexp(dim, keepdim));
}


// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_mm_ (Rcpp::XPtr<torch::Tensor> x, Rcpp::XPtr<torch::Tensor> mat2) {
  return make_tensor_ptr(x->mm(*mat2));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_mul_tensor_ (Rcpp::XPtr<torch::Tensor> x,
                                              Rcpp::XPtr<torch::Tensor> other) {
  return make_tensor_ptr(x->mul(*other));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_mul_scalar_ (Rcpp::XPtr<torch::Tensor> x,
                                              SEXP other) {
  return make_tensor_ptr(x->mul(scalar_from_r_(other)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_permute_ (Rcpp::XPtr<torch::Tensor> x,
                                           std::vector<std::int64_t> dims) {
  // TODO handle scalar multiplication
  return make_tensor_ptr(x->permute(dims));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_pow_tensor_ (Rcpp::XPtr<torch::Tensor> x,
                                              Rcpp::XPtr<torch::Tensor> exponent) {
  return make_tensor_ptr(x->pow(*exponent));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_pow_scalar_ (Rcpp::XPtr<torch::Tensor> x,
                                              SEXP exponent) {
  return make_tensor_ptr(x->pow(scalar_from_r_(exponent)));
}

// [[Rcpp::export]]
Rcpp::List tensor_qr_ (Rcpp::XPtr<torch::Tensor> x) {
  auto out = x->qr();
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_round_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->round());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_rsqrt_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->rsqrt());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_rsqrt__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->rsqrt_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sigmoid_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sigmoid());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sigmoid__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sigmoid_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sign_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sign());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sqrt_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sqrt());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sqrt__ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->sqrt_());
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sub_tensor_ (Rcpp::XPtr<torch::Tensor> x,
                                              Rcpp::XPtr<torch::Tensor> other,
                                              SEXP alpha) {
  return make_tensor_ptr(x->sub(*other, scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sub_scalar_ (Rcpp::XPtr<torch::Tensor> x,
                                              SEXP other,
                                              SEXP alpha) {

  return make_tensor_ptr(x->sub(scalar_from_r_(other), scalar_from_r_(alpha)));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sub_tensor__ (Rcpp::XPtr<torch::Tensor> x,
                                               Rcpp::XPtr<torch::Tensor> other,
                                               SEXP alpha) {
  x->sub_(*other, scalar_from_r_(alpha));
  return x;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sub_scalar__ (Rcpp::XPtr<torch::Tensor> x,
                                               SEXP other,
                                               SEXP alpha) {
  x->sub_(scalar_from_r_(other), scalar_from_r_(alpha));
  return x;
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_sum_ (Rcpp::XPtr<torch::Tensor> x,
                                       Rcpp::Nullable<std::vector<std::int64_t>> dim,
                                       bool keepdim = false
) {

  if (dim.isNull()) {
    return make_tensor_ptr(x->sum());
  } else if (dim.isNotNull()) {
    return make_tensor_ptr(x->sum(Rcpp::as<std::vector<std::int64_t>>(dim), keepdim));
  }

  Rcpp::stop("Not yet implemented");
}


// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_t_ (Rcpp::XPtr<torch::Tensor> x) {
  return make_tensor_ptr(x->t());
}


// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_tril_ (Rcpp::XPtr<torch::Tensor> x,
                                        std::int64_t diagonal) {
  return make_tensor_ptr(x->tril(diagonal));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_triu_ (Rcpp::XPtr<torch::Tensor> x,
                                        std::int64_t diagonal) {
  return make_tensor_ptr(x->triu(diagonal));
}



// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_to_ (Rcpp::XPtr<torch::Tensor> x,
                                      Rcpp::Nullable<std::string> dtype,
                                      Rcpp::Nullable<std::string> device,
                                      bool requires_grad) {

  torch::Tensor tensor = x->clone();

  if (dtype.isNotNull() & device.isNotNull()) {
    tensor = tensor.to(
      device_from_string(Rcpp::as<std::string>(device)),
      scalar_type_from_string(Rcpp::as<std::string>(dtype))
    );
  } else if (dtype.isNotNull()) {
    tensor = tensor.to(scalar_type_from_string(Rcpp::as<std::string>(dtype)));
  } else if (device.isNotNull()) {
    tensor = tensor.to(device_from_string(Rcpp::as<std::string>(device)));
  }

  if (requires_grad) {
    tensor = tensor.set_requires_grad(requires_grad);
  }

  return make_tensor_ptr(tensor);
}

// [[Rcpp::export]]
Rcpp::List tensor_unique_return_inverse_ (Rcpp::XPtr<torch::Tensor> x,
                                          bool sorted,
                                          Rcpp::Nullable<std::int64_t> dim) {
  if(dim.isNull()) {
    auto out = at::_unique(*x, sorted, true);
    return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
  } else {
    auto out = at::_unique_dim(*x, Rcpp::as<std::int64_t>(dim), sorted, true);
    return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
  }
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_unique_ (Rcpp::XPtr<torch::Tensor> x,
                                          bool sorted,
                                          Rcpp::Nullable<std::int64_t> dim) {
  if(dim.isNull()) {
    auto out = at::_unique(*x, sorted, false);
    return make_tensor_ptr(std::get<0>(out));
  } else {
    auto out = at::_unique_dim(*x, Rcpp::as<std::int64_t>(dim), sorted, false);
    return make_tensor_ptr(std::get<0>(out));
  }
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_unsqueeze_ (Rcpp::XPtr<torch::Tensor> x,
                                             std::int64_t dim) {
  return make_tensor_ptr(x->unsqueeze(dim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_unsqueeze__ (Rcpp::XPtr<torch::Tensor> x,
                                              std::int64_t dim) {
  return make_tensor_ptr(x->unsqueeze_(dim));
}

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_zero__ (Rcpp::XPtr<torch::Tensor> x) {
  x->zero_();
  return x;
}
