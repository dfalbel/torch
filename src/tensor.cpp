#include "torch_types.h"

RTensor make_tensor_ptr (torch::Tensor x) {
  auto * out = new torch::Tensor(x);
  return RTensor(out);
}

std::vector<int64_t> reverse_int_seq (int n) {
  std::vector<int64_t> l(n);
  std::iota(l.begin(), l.end(), 0);
  std::reverse(l.begin(), l.end());
  return l;
};

template <int RTYPE, at::ScalarType ATTYPE>
RTensor tensor_from_r_impl_ (const SEXP x, const std::vector<int64_t> dim) {

  Rcpp::Vector<RTYPE> vec(x);

  auto tensor = torch::from_blob(vec.begin(), dim, ATTYPE);

  if (dim.size() == 1) {
    // if we have a 1-dim vector contigous doesn't trigger a copy, and
    // would be unexpected.
    tensor = tensor.clone();
  }

  tensor = tensor
    .permute(reverse_int_seq(dim.size()))
    .contiguous(); // triggers a copy!

  if (RTYPE == LGLSXP)
    tensor = tensor.to(torch::kByte);

  return make_tensor_ptr(tensor);
};

// [[Rcpp::export]]
RTensor tensor_from_r_ (SEXP x, std::vector<int64_t> dim) {

  switch (TYPEOF(x)) {
  case INTSXP:
    return tensor_from_r_impl_<INTSXP, torch::kInt>(x, dim);
  case REALSXP:
    return tensor_from_r_impl_<REALSXP, torch::kDouble>(x, dim);
  case LGLSXP:
    // since R logical vectors have 8B we need to treat them as integer vectors
    // and then cast to bit tensor.
    return tensor_from_r_impl_<LGLSXP, torch::kInt32>(x, dim);
  default:
    Rcpp::stop("not handled");
  }
};

torch::ScalarType scalar_type_from_string(std::string scalar_type) {
  if (scalar_type == "kInt") {
    return torch::kInt;
  } else if (scalar_type == "kDouble") {
    return torch::kDouble;
  }
  Rcpp::stop("scalar not handled");
}

torch::Device device_from_string(std::string device) {
  if (device == "CPU") {
    return torch::Device(torch::DeviceType::CPU);
  }
  Rcpp::stop("device not handled");
}

// [[Rcpp::export]]
RTensor tensor_ (RTensor x,
                 Rcpp::Nullable<Rcpp::CharacterVector> dtype,
                 Rcpp::Nullable<Rcpp::CharacterVector> device,
                 bool requires_grad) {
  if (dtype.isNull() & device.isNull()) {
    if (x->requires_grad() == requires_grad) {
      return make_tensor_ptr(*x);
    } else {
      auto out = x->clone();
      out.set_requires_grad(requires_grad);
      return make_tensor_ptr(out);
    }
  } else if (dtype.isNull() & device.isNotNull()) {
    auto out = x->to(device_from_string(Rcpp::as<std::string>(device)));
    out.set_requires_grad(requires_grad);
    return make_tensor_ptr(out);
  } else if (dtype.isNotNull() & device.isNull()) {
    auto out = x->to(scalar_type_from_string(Rcpp::as<std::string>(dtype)));
    out.set_requires_grad(requires_grad);
    return make_tensor_ptr(out);
  } else if (dtype.isNotNull() & device.isNotNull()) {
    auto out = x->to(
      device_from_string(Rcpp::as<std::string>(device)),
      scalar_type_from_string(Rcpp::as<std::string>(dtype))
    );
    out.set_requires_grad(requires_grad);
    return make_tensor_ptr(out);
  }
  Rcpp::stop("not handled");
}

// [[Rcpp::export]]
void tensor_print_ (RTensor x) {
  torch::Tensor ten = *x;
  Rcpp::Rcout << ten << std::endl;
};

template <int RTYPE, typename STDTYPE>
Rcpp::List as_array_tensor_impl_ (RTensor x) {

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
Rcpp::List as_array_tensor_ (RTensor x) {

  torch::Tensor ten = *x;

  if (ten.dtype() == torch::kInt) {
    return as_array_tensor_impl_<INTSXP, int32_t>(x);
  } else if (ten.dtype() == torch::kDouble) {
    return as_array_tensor_impl_<REALSXP, double>(x);
  } else if (ten.dtype() == torch::kByte) {
    // TODO:
    // not sure why this works :(
    return as_array_tensor_impl_<LGLSXP, std::uint8_t>(x);
  } else if (ten.dtype() == torch::kLong) {
    // TODO: deal better with kLongs
    // Klong is casted to kInt inside impl
    return as_array_tensor_impl_<INTSXP, int32_t>(x);
  }

  Rcpp::stop("not handled");
};

// [[Rcpp::export]]
RTensor tensor_abs_ (RTensor x) {
  return make_tensor_ptr(x->abs());
}

// [[Rcpp::export]]
RTensor tensor_acos_ (RTensor x) {
  return make_tensor_ptr(x->acos());
}

// [[Rcpp::export]]
RTensor tensor_add_ (RTensor x, RTensor y) {
  return make_tensor_ptr(x->add(*y));
}

// [[Rcpp::export]]
RTensor tensor_addbmm_ (RTensor x, RTensor batch1,
                        RTensor batch2, double beta, double alpha) {
  return make_tensor_ptr(x->addbmm(*batch1, *batch2, beta, alpha));
}

// [[Rcpp::export]]
RTensor tensor_addcdiv_ (RTensor x, RTensor tensor1,
                         RTensor tensor2, double value
) {
  return make_tensor_ptr(x->addcdiv(*tensor1, *tensor2, value));
}

// [[Rcpp::export]]
RTensor tensor_addcmul_ (RTensor x, RTensor tensor1,
                         RTensor tensor2, double value
) {
  return make_tensor_ptr(x->addcmul(*tensor1, *tensor2, value));
}

// [[Rcpp::export]]
RTensor tensor_addmm_ (RTensor x, RTensor mat1,
                       RTensor mat2, double beta, double alpha
) {
  return make_tensor_ptr(x->addmm(*mat1, *mat2, beta, alpha));
}

// [[Rcpp::export]]
RTensor tensor_addmv_ (RTensor x, RTensor mat,
                       RTensor vec, double beta, double alpha) {
  return make_tensor_ptr(x->addmv(*mat, *vec, beta, alpha));
}

// [[Rcpp::export]]
RTensor tensor_addr_ (RTensor x, RTensor vec1,
                      RTensor vec2, double beta, double alpha) {
  return make_tensor_ptr(x->addr(*vec1, *vec2, beta, alpha));
}

// [[Rcpp::export]]
RTensor tensor_all_ (RTensor x,
                     Rcpp::Nullable<Rcpp::IntegerVector> dim,
                     bool keepdim) {

  if (dim.isNull())
    return make_tensor_ptr(x->all());
  else {
    std::int64_t dim2 = Rcpp::as<int64_t>(dim);
    return make_tensor_ptr(x->all(dim2, keepdim));
  }
}

// [[Rcpp::export]]
bool tensor_allclose_ (RTensor x, RTensor other,
                       double rtol, double atol, bool equal_nan) {
  return x->allclose(*other, rtol, atol, equal_nan);
}

// [[Rcpp::export]]
RTensor tensor_any_ (RTensor x,
                     Rcpp::Nullable<Rcpp::IntegerVector> dim,
                     bool keepdim) {
  if (dim.isNull())
    return make_tensor_ptr(x->any());
  else {
    std::int64_t dim2 = Rcpp::as<int64_t>(dim);
    return make_tensor_ptr(x->any(dim2, keepdim));
  }
}

// [[Rcpp::export]]
RTensor tensor_argmax_ (RTensor x,
                        Rcpp::Nullable<Rcpp::IntegerVector> dim,
                        bool keepdim) {
  if (dim.isNull())
    return make_tensor_ptr(x->argmax());
  else {
    std::int64_t dim2 = Rcpp::as<int64_t>(dim);
    return make_tensor_ptr(x->argmax(dim2, keepdim));
  }
}

// [[Rcpp::export]]
RTensor tensor_argmin_ (RTensor x,
                        Rcpp::Nullable<Rcpp::IntegerVector> dim,
                        bool keepdim) {
  if (dim.isNull())
    return make_tensor_ptr(x->argmin());
  else {
    std::int64_t dim2 = Rcpp::as<int64_t>(dim);
    return make_tensor_ptr(x->argmin(dim2, keepdim));
  }
}

// [[Rcpp::export]]
RTensor tensor_as_strided_ (RTensor x,
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
std::string tensor_to_string_ (RTensor x) {
  torch::Tensor ten = *x;
  return ten.toString();
}

// [[Rcpp::export]]
RTensor tensor_asin_ (RTensor x) {
  return make_tensor_ptr(x->asin());
}

// [[Rcpp::export]]
RTensor tensor_atan_ (RTensor x) {
  return make_tensor_ptr(x->atan());
}

// [[Rcpp::export]]
RTensor tensor_atan2_ (RTensor x, RTensor other) {
  return make_tensor_ptr(x->atan2(*other));
}

// [[Rcpp::export]]
void tensor_backward_ (RTensor x,
                       Rcpp::Nullable<RTensor> gradient,
                       bool keep_graph, bool create_graph) {
  if(gradient.isNull()) {
    x->backward(at::nullopt, keep_graph, create_graph);
  } else {
    RTensor gradient2(gradient.get());
    x->backward(*gradient2, keep_graph, create_graph);
  }

}

// [[Rcpp::export]]
RTensor tensor_baddbmm_ (RTensor x, RTensor batch1,
                         RTensor batch2, double beta, double alpha) {
  return make_tensor_ptr(x->baddbmm(*batch1, *batch2, beta, alpha));
}

// [[Rcpp::export]]
RTensor tensor_bernoulli_ (RTensor x,
                           Rcpp::Nullable<Rcpp::NumericVector> p) {
  if (p.isNull()) {
    return make_tensor_ptr(x->bernoulli());
  } else {
    double p2 = Rcpp::as<double>(p.get());
    return make_tensor_ptr(x->bernoulli(p2));
  }
}

// [[Rcpp::export]]
RTensor tensor_bincount_ (RTensor x,
                          Rcpp::Nullable<RTensor> weights,
                          std::int64_t minlength) {

  if (weights.isNull()) {
    return make_tensor_ptr(x->bincount({}, minlength));
  } else {
    return make_tensor_ptr(x->bincount(*(RTensor(weights.get())), minlength));
  }
}

// [[Rcpp::export]]
RTensor tensor_bmm_ (RTensor x,
                     RTensor mat2) {
  return make_tensor_ptr(x->bmm(*mat2));
}

// [[Rcpp::export]]
Rcpp::List tensor_btrifact_ (RTensor x, bool pivot) {
  auto out = x->btrifact(pivot);
  return Rcpp::List::create(make_tensor_ptr(std::get<0>(out)), make_tensor_ptr(std::get<1>(out)));
}

// [[Rcpp::export]]
RTensor tensor_btrisolve_ (RTensor x,
                           RTensor LU_data,
                           RTensor LU_pivots) {
  return make_tensor_ptr(x->btrisolve(*LU_data, *LU_pivots));
}

// [[Rcpp::export]]
RTensor tensor_cauchy__ (RTensor x, double median, double sigma) {
  return make_tensor_ptr(x->cauchy_(median, sigma));
}

// [[Rcpp::export]]
RTensor tensor_ceil_ (RTensor x) {
  return make_tensor_ptr(x->ceil());
}

// [[Rcpp::export]]
RTensor tensor_ceil__ (RTensor x) {
  return make_tensor_ptr(x->ceil_());
}

// [[Rcpp::export]]
Rcpp::List tensor_chunk_ (RTensor x, int64_t chunks, int64_t dim) {

  auto chunks_vector = x->chunk(chunks, dim);
  Rcpp::List out;

  for (int i = 0; i < chunks_vector.size(); i++) {
    out.push_back(make_tensor_ptr(chunks_vector[i]));
  }

  return out;
}

// [[Rcpp::export]]
RTensor tensor_clamp_ (RTensor x,
                       double min,
                       double max) {
  return make_tensor_ptr(x->clamp(min, max));
}

// [[Rcpp::export]]
RTensor tensor_clamp__ (RTensor x,
                        double min,
                        double max) {
  return make_tensor_ptr(x->clamp_(min, max));
}

// [[Rcpp::export]]
RTensor tensor_clamp_max_ (RTensor x, double max) {
  return make_tensor_ptr(x->clamp_max(max));
}

// [[Rcpp::export]]
RTensor tensor_clamp_max__ (RTensor x, double max) {
  return make_tensor_ptr(x->clamp_max_(max));
}

// [[Rcpp::export]]
RTensor tensor_clamp_min_ (RTensor x, double min) {
  return make_tensor_ptr(x->clamp_min(min));
}

// [[Rcpp::export]]
RTensor tensor_clamp_min__ (RTensor x, double min) {
  return make_tensor_ptr(x->clamp_min_(min));
}

// [[Rcpp::export]]
RTensor tensor_clone_ (RTensor x) {
  return make_tensor_ptr(x->clone());
}

// [[Rcpp::export]]
RTensor tensor_contiguous_ (RTensor x) {
  return make_tensor_ptr(x->contiguous());
}

// [[Rcpp::export]]
RTensor tensor_copy__ (RTensor x,
                       RTensor src,
                       bool non_blocking = false) {
  return make_tensor_ptr(x->copy_(*src, non_blocking));
}

// [[Rcpp::export]]
RTensor tensor_data_ (RTensor x) {
  auto out = torch::from_blob(x->data_ptr(), x->sizes(), x->type());
  return make_tensor_ptr(out);
}

// [[Rcpp::export]]
RTensor tensor_grad_ (RTensor x) {
  return make_tensor_ptr(x->grad());
}

// [[Rcpp::export]]
RTensor tensor_mean_ (RTensor x,
                      Rcpp::Nullable<Rcpp::IntegerVector> dim,
                      Rcpp::Nullable<Rcpp::LogicalVector> keepdim,
                      Rcpp::Nullable<Rcpp::CharacterVector> dtype) {

  if (dim.isNull() & keepdim.isNull() & dtype.isNull()) {
    return make_tensor_ptr(x->mean());
  }

  // TODO handle other sum arguments.
  Rcpp::stop("Not yet implemented");
}


// [[Rcpp::export]]
RTensor tensor_mm_ (RTensor x, RTensor mat2) {
  return make_tensor_ptr(x->mm(*mat2));
}

// [[Rcpp::export]]
RTensor tensor_mul_ (RTensor x,
                     RTensor other) {
  // TODO handle scalar multiplication
  return make_tensor_ptr(x->mul(*other));
}

// [[Rcpp::export]]
RTensor tensor_permute_ (RTensor x,
                         std::vector<std::int64_t> dims) {
  // TODO handle scalar multiplication
  return make_tensor_ptr(x->permute(dims));
}

// [[Rcpp::export]]
RTensor tensor_pow_ (RTensor x,
                     RTensor exponent) {
  // TODO handle scalar multiplication
  return make_tensor_ptr(x->pow(*exponent));
}

// [[Rcpp::export]]
RTensor tensor_sub_ (RTensor x,
                     RTensor other,
                     double alpha = 1) {
  return make_tensor_ptr(x->sub(*other, alpha));
}

// [[Rcpp::export]]
RTensor tensor_sub__ (RTensor x,
                      RTensor other,
                      double alpha = 1) {
  x->sub_(*other, alpha);
  return x;
}

// [[Rcpp::export]]
RTensor tensor_sum_ (RTensor x,
                     Rcpp::Nullable<Rcpp::IntegerVector> dim,
                     Rcpp::Nullable<Rcpp::LogicalVector> keepdim,
                     Rcpp::Nullable<Rcpp::CharacterVector> dtype) {

  if (dim.isNull() & keepdim.isNull() & dtype.isNull()) {
    return make_tensor_ptr(x->sum());
  }

  // TODO handle other sum arguments.
  Rcpp::stop("Not yet implemented");
}


// [[Rcpp::export]]
RTensor tensor_t_ (RTensor x) {
  return make_tensor_ptr(x->t());
}

// [[Rcpp::export]]
RTensor tensor_zero__ (RTensor x) {
  x->zero_();
  return x;
}
