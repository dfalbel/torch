#include "torch_types.h"
using namespace Rcpp;

// [[Rcpp::export]]
int hello2() {
  torch::Tensor tensor = torch::rand({11, 11, 3});
  auto ten = std::make_shared<torch::Tensor>(tensor);
  Rcout << *ten << std::endl;
  return 1;
};


// [[Rcpp::export]]
SEXP hello3 () {

  torch::Tensor tensor = torch::rand({11, 11, 3});
  auto * ten = new torch::Tensor(tensor);

  Rcout << *ten << std::endl;

  auto ptr = Rcpp::XPtr<torch::Tensor>(ten);
  return ptr;
};

// [[Rcpp::export]]
void print_tensor (SEXP a) {
  Rcpp::XPtr<torch::Tensor> x(a);
  torch::Tensor b = *x;
  Rcout << b << std::endl;
}
