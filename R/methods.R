setGeneric("torch_backward_", function(self, gradient, keep_graph, create_graph) standardGeneric("torch_backward_"))
setGeneric("torch_set_data_", function(self, new_data) standardGeneric("torch_set_data_"))
setGeneric("torch_abs_", function(self) standardGeneric("torch_abs_"))
setGeneric("torch_abs__", function(self) standardGeneric("torch_abs__"))
setGeneric("torch_acos_", function(self) standardGeneric("torch_acos_"))
setGeneric("torch_acos__", function(self) standardGeneric("torch_acos__"))
setGeneric("torch_add_", function(self, other, alpha) standardGeneric("torch_add_"))
setGeneric("torch_add__", function(self, other, alpha) standardGeneric("torch_add__"))
setGeneric("torch_addmv_", function(self, mat, vec, beta, alpha) standardGeneric("torch_addmv_"))
setGeneric("torch_addmv__", function(self, mat, vec, beta, alpha) standardGeneric("torch_addmv__"))
setGeneric("torch_addr_", function(self, vec1, vec2, beta, alpha) standardGeneric("torch_addr_"))
setGeneric("torch_addr__", function(self, vec1, vec2, beta, alpha) standardGeneric("torch_addr__"))
setGeneric("torch_all_", function(self, dim, keepdim) standardGeneric("torch_all_"))
setGeneric("torch_allclose_", function(self, other, rtol, atol, equal_nan) standardGeneric("torch_allclose_"))
setGeneric("torch_any_", function(self, dim, keepdim) standardGeneric("torch_any_"))
setGeneric("torch_argmax_", function(self, dim, keepdim) standardGeneric("torch_argmax_"))
setGeneric("torch_argmin_", function(self, dim, keepdim) standardGeneric("torch_argmin_"))
setGeneric("torch_as_strided_", function(self, size, stride, storage_offset) standardGeneric("torch_as_strided_"))
setGeneric("torch_as_strided__", function(self, size, stride, storage_offset) standardGeneric("torch_as_strided__"))
setGeneric("torch_asin_", function(self) standardGeneric("torch_asin_"))
setGeneric("torch_asin__", function(self) standardGeneric("torch_asin__"))
setGeneric("torch_atan_", function(self) standardGeneric("torch_atan_"))
setGeneric("torch_atan__", function(self) standardGeneric("torch_atan__"))
setGeneric("torch_baddbmm_", function(self, batch1, batch2, beta, alpha) standardGeneric("torch_baddbmm_"))
setGeneric("torch_baddbmm__", function(self, batch1, batch2, beta, alpha) standardGeneric("torch_baddbmm__"))
setGeneric("torch_bernoulli_", function(self, p) standardGeneric("torch_bernoulli_"))
setGeneric("torch_bernoulli__", function(self, p) standardGeneric("torch_bernoulli__"))
setGeneric("torch_bincount_", function(self, weights, minlength) standardGeneric("torch_bincount_"))
setGeneric("torch_bitwise_not_", function(self) standardGeneric("torch_bitwise_not_"))
setGeneric("torch_bitwise_not__", function(self) standardGeneric("torch_bitwise_not__"))
setGeneric("torch_bmm_", function(self, mat2) standardGeneric("torch_bmm_"))
setGeneric("torch_ceil_", function(self) standardGeneric("torch_ceil_"))
setGeneric("torch_ceil__", function(self) standardGeneric("torch_ceil__"))
setGeneric("torch_chunk_", function(self, chunks, dim) standardGeneric("torch_chunk_"))
setGeneric("torch_clamp_", function(self, min, max) standardGeneric("torch_clamp_"))
setGeneric("torch_clamp__", function(self, min, max) standardGeneric("torch_clamp__"))
setGeneric("torch_clamp_max_", function(self, max) standardGeneric("torch_clamp_max_"))
setGeneric("torch_clamp_max__", function(self, max) standardGeneric("torch_clamp_max__"))
setGeneric("torch_clamp_min_", function(self, min) standardGeneric("torch_clamp_min_"))
setGeneric("torch_clamp_min__", function(self, min) standardGeneric("torch_clamp_min__"))
setGeneric("torch_contiguous_", function(self, memory_format) standardGeneric("torch_contiguous_"))
setGeneric("torch_copy__", function(self, src, non_blocking) standardGeneric("torch_copy__"))
setGeneric("torch_cos_", function(self) standardGeneric("torch_cos_"))
setGeneric("torch_cos__", function(self) standardGeneric("torch_cos__"))
setGeneric("torch_cosh_", function(self) standardGeneric("torch_cosh_"))
setGeneric("torch_cosh__", function(self) standardGeneric("torch_cosh__"))
setGeneric("torch_cumsum_", function(self, dim, dtype) standardGeneric("torch_cumsum_"))
setGeneric("torch_cumprod_", function(self, dim, dtype) standardGeneric("torch_cumprod_"))
setGeneric("torch_det_", function(self) standardGeneric("torch_det_"))
setGeneric("torch_diag_embed_", function(self, offset, dim1, dim2) standardGeneric("torch_diag_embed_"))
setGeneric("torch_diagflat_", function(self, offset) standardGeneric("torch_diagflat_"))
setGeneric("torch_diagonal_", function(self, offset, dim1, dim2) standardGeneric("torch_diagonal_"))
setGeneric("torch_fill_diagonal__", function(self, fill_value, wrap) standardGeneric("torch_fill_diagonal__"))
setGeneric("torch_div_", function(self, other) standardGeneric("torch_div_"))
setGeneric("torch_div__", function(self, other) standardGeneric("torch_div__"))
setGeneric("torch_dot_", function(self, tensor) standardGeneric("torch_dot_"))
setGeneric("torch_resize__", function(self, size) standardGeneric("torch_resize__"))
setGeneric("torch_erf_", function(self) standardGeneric("torch_erf_"))
setGeneric("torch_erf__", function(self) standardGeneric("torch_erf__"))
setGeneric("torch_erfc_", function(self) standardGeneric("torch_erfc_"))
setGeneric("torch_erfc__", function(self) standardGeneric("torch_erfc__"))
setGeneric("torch_exp_", function(self) standardGeneric("torch_exp_"))
setGeneric("torch_exp__", function(self) standardGeneric("torch_exp__"))
setGeneric("torch_expm1_", function(self) standardGeneric("torch_expm1_"))
setGeneric("torch_expm1__", function(self) standardGeneric("torch_expm1__"))
setGeneric("torch_expand_", function(self, size, implicit) standardGeneric("torch_expand_"))
setGeneric("torch_expand_as_", function(self, other) standardGeneric("torch_expand_as_"))
setGeneric("torch_flatten_", function(self, start_dim, end_dim) standardGeneric("torch_flatten_"))
setGeneric("torch_fill__", function(self, value) standardGeneric("torch_fill__"))
setGeneric("torch_floor_", function(self) standardGeneric("torch_floor_"))
setGeneric("torch_floor__", function(self) standardGeneric("torch_floor__"))
setGeneric("torch_frac_", function(self) standardGeneric("torch_frac_"))
setGeneric("torch_frac__", function(self) standardGeneric("torch_frac__"))
setGeneric("torch_ger_", function(self, vec2) standardGeneric("torch_ger_"))
setGeneric("torch_fft_", function(self, signal_ndim, normalized) standardGeneric("torch_fft_"))
setGeneric("torch_ifft_", function(self, signal_ndim, normalized) standardGeneric("torch_ifft_"))
setGeneric("torch_rfft_", function(self, signal_ndim, normalized, onesided) standardGeneric("torch_rfft_"))
setGeneric("torch_irfft_", function(self, signal_ndim, normalized, onesided, signal_sizes) standardGeneric("torch_irfft_"))
setGeneric("torch_index_", function(self, indices) standardGeneric("torch_index_"))
setGeneric("torch_index_copy__", function(self, dim, index, source) standardGeneric("torch_index_copy__"))
setGeneric("torch_index_copy_", function(self, dim, index, source) standardGeneric("torch_index_copy_"))
setGeneric("torch_index_put__", function(self, indices, values, accumulate) standardGeneric("torch_index_put__"))
setGeneric("torch_index_put_", function(self, indices, values, accumulate) standardGeneric("torch_index_put_"))
setGeneric("torch_inverse_", function(self) standardGeneric("torch_inverse_"))
setGeneric("torch_isclose_", function(self, other, rtol, atol, equal_nan) standardGeneric("torch_isclose_"))
setGeneric("torch_is_distributed_", function(self) standardGeneric("torch_is_distributed_"))
setGeneric("torch_is_floating_point_", function(self) standardGeneric("torch_is_floating_point_"))
setGeneric("torch_is_complex_", function(self) standardGeneric("torch_is_complex_"))
setGeneric("torch_is_nonzero_", function(self) standardGeneric("torch_is_nonzero_"))
setGeneric("torch_is_same_size_", function(self, other) standardGeneric("torch_is_same_size_"))
setGeneric("torch_is_signed_", function(self) standardGeneric("torch_is_signed_"))
setGeneric("torch_kthvalue_", function(self, k, dim, keepdim) standardGeneric("torch_kthvalue_"))
setGeneric("torch_log_", function(self) standardGeneric("torch_log_"))
setGeneric("torch_log__", function(self) standardGeneric("torch_log__"))
setGeneric("torch_log10_", function(self) standardGeneric("torch_log10_"))
setGeneric("torch_log10__", function(self) standardGeneric("torch_log10__"))
setGeneric("torch_log1p_", function(self) standardGeneric("torch_log1p_"))
setGeneric("torch_log1p__", function(self) standardGeneric("torch_log1p__"))
setGeneric("torch_log2_", function(self) standardGeneric("torch_log2_"))
setGeneric("torch_log2__", function(self) standardGeneric("torch_log2__"))
setGeneric("torch_logdet_", function(self) standardGeneric("torch_logdet_"))
setGeneric("torch_log_softmax_", function(self, dim, dtype) standardGeneric("torch_log_softmax_"))
setGeneric("torch_logsumexp_", function(self, dim, keepdim) standardGeneric("torch_logsumexp_"))
setGeneric("torch_matmul_", function(self, other) standardGeneric("torch_matmul_"))
setGeneric("torch_matrix_power_", function(self, n) standardGeneric("torch_matrix_power_"))
setGeneric("torch_max_", function(self, dim, other, keepdim) standardGeneric("torch_max_"))
setGeneric("torch_max_values_", function(self, dim, keepdim) standardGeneric("torch_max_values_"))
setGeneric("torch_mean_", function(self, dim, keepdim, dtype) standardGeneric("torch_mean_"))
setGeneric("torch_median_", function(self, dim, keepdim) standardGeneric("torch_median_"))
setGeneric("torch_min_", function(self, dim, other, keepdim) standardGeneric("torch_min_"))
setGeneric("torch_min_values_", function(self, dim, keepdim) standardGeneric("torch_min_values_"))
setGeneric("torch_mm_", function(self, mat2) standardGeneric("torch_mm_"))
setGeneric("torch_mode_", function(self, dim, keepdim) standardGeneric("torch_mode_"))
setGeneric("torch_mul_", function(self, other) standardGeneric("torch_mul_"))
setGeneric("torch_mul__", function(self, other) standardGeneric("torch_mul__"))
setGeneric("torch_mv_", function(self, vec) standardGeneric("torch_mv_"))
setGeneric("torch_mvlgamma_", function(self, p) standardGeneric("torch_mvlgamma_"))
setGeneric("torch_mvlgamma__", function(self, p) standardGeneric("torch_mvlgamma__"))
setGeneric("torch_narrow_copy_", function(self, dim, start, length) standardGeneric("torch_narrow_copy_"))
setGeneric("torch_narrow_", function(self, dim, start, length) standardGeneric("torch_narrow_"))
setGeneric("torch_permute_", function(self, dims) standardGeneric("torch_permute_"))
setGeneric("torch_numpy_T_", function(self) standardGeneric("torch_numpy_T_"))
setGeneric("torch_pin_memory_", function(self) standardGeneric("torch_pin_memory_"))
setGeneric("torch_pinverse_", function(self, rcond) standardGeneric("torch_pinverse_"))
setGeneric("torch_reciprocal_", function(self) standardGeneric("torch_reciprocal_"))
setGeneric("torch_reciprocal__", function(self) standardGeneric("torch_reciprocal__"))
setGeneric("torch_neg_", function(self) standardGeneric("torch_neg_"))
setGeneric("torch_neg__", function(self) standardGeneric("torch_neg__"))
setGeneric("torch_repeat_", function(self, repeats) standardGeneric("torch_repeat_"))
setGeneric("torch_repeat_interleave_", function(self, repeats, dim) standardGeneric("torch_repeat_interleave_"))
setGeneric("torch_reshape_", function(self, shape) standardGeneric("torch_reshape_"))
setGeneric("torch_reshape_as_", function(self, other) standardGeneric("torch_reshape_as_"))
setGeneric("torch_round_", function(self) standardGeneric("torch_round_"))
setGeneric("torch_round__", function(self) standardGeneric("torch_round__"))
setGeneric("torch_relu_", function(self) standardGeneric("torch_relu_"))
setGeneric("torch_relu__", function(self) standardGeneric("torch_relu__"))
setGeneric("torch_prelu_", function(self, weight) standardGeneric("torch_prelu_"))
setGeneric("torch_prelu_backward_", function(grad_output, self, weight) standardGeneric("torch_prelu_backward_"))
setGeneric("torch_hardshrink_", function(self, lambd) standardGeneric("torch_hardshrink_"))
setGeneric("torch_hardshrink_backward_", function(grad_out, self, lambd) standardGeneric("torch_hardshrink_backward_"))
setGeneric("torch_rsqrt_", function(self) standardGeneric("torch_rsqrt_"))
setGeneric("torch_rsqrt__", function(self) standardGeneric("torch_rsqrt__"))
setGeneric("torch_select_", function(self, dim, index) standardGeneric("torch_select_"))
setGeneric("torch_sigmoid_", function(self) standardGeneric("torch_sigmoid_"))
setGeneric("torch_sigmoid__", function(self) standardGeneric("torch_sigmoid__"))
setGeneric("torch_sin_", function(self) standardGeneric("torch_sin_"))
setGeneric("torch_sin__", function(self) standardGeneric("torch_sin__"))
setGeneric("torch_sinh_", function(self) standardGeneric("torch_sinh_"))
setGeneric("torch_sinh__", function(self) standardGeneric("torch_sinh__"))
setGeneric("torch_detach_", function(self) standardGeneric("torch_detach_"))
setGeneric("torch_detach__", function(self) standardGeneric("torch_detach__"))
setGeneric("torch_size_", function(self, dim) standardGeneric("torch_size_"))
setGeneric("torch_slice_", function(self, dim, start, end, step) standardGeneric("torch_slice_"))
setGeneric("torch_slogdet_", function(self) standardGeneric("torch_slogdet_"))
setGeneric("torch_smm_", function(self, mat2) standardGeneric("torch_smm_"))
setGeneric("torch_softmax_", function(self, dim, dtype) standardGeneric("torch_softmax_"))
setGeneric("torch_split_", function(self, split_size, dim) standardGeneric("torch_split_"))
setGeneric("torch_split_with_sizes_", function(self, split_sizes, dim) standardGeneric("torch_split_with_sizes_"))
setGeneric("torch_squeeze_", function(self, dim) standardGeneric("torch_squeeze_"))
setGeneric("torch_squeeze__", function(self, dim) standardGeneric("torch_squeeze__"))
setGeneric("torch_sspaddmm_", function(self, mat1, mat2, beta, alpha) standardGeneric("torch_sspaddmm_"))
setGeneric("torch_stft_", function(self, n_fft, hop_length, win_length, window, normalized, onesided) standardGeneric("torch_stft_"))
setGeneric("torch_stride_", function(self, dim) standardGeneric("torch_stride_"))
setGeneric("torch_sum_", function(self, dim, keepdim, dtype) standardGeneric("torch_sum_"))
setGeneric("torch_sum_to_size_", function(self, size) standardGeneric("torch_sum_to_size_"))
setGeneric("torch_sqrt_", function(self) standardGeneric("torch_sqrt_"))
setGeneric("torch_sqrt__", function(self) standardGeneric("torch_sqrt__"))
setGeneric("torch_std_", function(self, dim, unbiased, keepdim) standardGeneric("torch_std_"))
setGeneric("torch_prod_", function(self, dim, keepdim, dtype) standardGeneric("torch_prod_"))
setGeneric("torch_t_", function(self) standardGeneric("torch_t_"))
setGeneric("torch_t__", function(self) standardGeneric("torch_t__"))
setGeneric("torch_tan_", function(self) standardGeneric("torch_tan_"))
setGeneric("torch_tan__", function(self) standardGeneric("torch_tan__"))
setGeneric("torch_tanh_", function(self) standardGeneric("torch_tanh_"))
setGeneric("torch_tanh__", function(self) standardGeneric("torch_tanh__"))
setGeneric("torch_transpose_", function(self, dim0, dim1) standardGeneric("torch_transpose_"))
setGeneric("torch_transpose__", function(self, dim0, dim1) standardGeneric("torch_transpose__"))
setGeneric("torch_flip_", function(self, dims) standardGeneric("torch_flip_"))
setGeneric("torch_roll_", function(self, shifts, dims) standardGeneric("torch_roll_"))
setGeneric("torch_rot90_", function(self, k, dims) standardGeneric("torch_rot90_"))
setGeneric("torch_trunc_", function(self) standardGeneric("torch_trunc_"))
setGeneric("torch_trunc__", function(self) standardGeneric("torch_trunc__"))
setGeneric("torch_type_as_", function(self, other) standardGeneric("torch_type_as_"))
setGeneric("torch_unsqueeze_", function(self, dim) standardGeneric("torch_unsqueeze_"))
setGeneric("torch_unsqueeze__", function(self, dim) standardGeneric("torch_unsqueeze__"))
setGeneric("torch_var_", function(self, dim, unbiased, keepdim) standardGeneric("torch_var_"))
setGeneric("torch_view_as_", function(self, other) standardGeneric("torch_view_as_"))
setGeneric("torch_where_", function(condition, self, other) standardGeneric("torch_where_"))
setGeneric("torch_norm_", function(self, p, dim, keepdim, dtype) standardGeneric("torch_norm_"))
setGeneric("torch_clone_", function(self) standardGeneric("torch_clone_"))
setGeneric("torch_resize_as__", function(self, the_template) standardGeneric("torch_resize_as__"))
setGeneric("torch_pow_", function(self, exponent) standardGeneric("torch_pow_"))
setGeneric("torch_zero__", function(self) standardGeneric("torch_zero__"))
setGeneric("torch_sub_", function(self, other, alpha) standardGeneric("torch_sub_"))
setGeneric("torch_sub__", function(self, other, alpha) standardGeneric("torch_sub__"))
setGeneric("torch_addmm_", function(self, mat1, mat2, beta, alpha) standardGeneric("torch_addmm_"))
setGeneric("torch_addmm__", function(self, mat1, mat2, beta, alpha) standardGeneric("torch_addmm__"))
setGeneric("torch_sparse_resize__", function(self, size, sparse_dim, dense_dim) standardGeneric("torch_sparse_resize__"))
setGeneric("torch_sparse_resize_and_clear__", function(self, size, sparse_dim, dense_dim) standardGeneric("torch_sparse_resize_and_clear__"))
setGeneric("torch_sparse_mask_", function(self, mask) standardGeneric("torch_sparse_mask_"))
setGeneric("torch_to_dense_", function(self) standardGeneric("torch_to_dense_"))
setGeneric("torch_sparse_dim_", function(self) standardGeneric("torch_sparse_dim_"))
setGeneric("torch__dimI_", function(self) standardGeneric("torch__dimI_"))
setGeneric("torch_dense_dim_", function(self) standardGeneric("torch_dense_dim_"))
setGeneric("torch__dimV_", function(self) standardGeneric("torch__dimV_"))
setGeneric("torch__nnz_", function(self) standardGeneric("torch__nnz_"))
setGeneric("torch_coalesce_", function(self) standardGeneric("torch_coalesce_"))
setGeneric("torch_is_coalesced_", function(self) standardGeneric("torch_is_coalesced_"))
setGeneric("torch__indices_", function(self) standardGeneric("torch__indices_"))
setGeneric("torch__values_", function(self) standardGeneric("torch__values_"))
setGeneric("torch__coalesced__", function(self, coalesced) standardGeneric("torch__coalesced__"))
setGeneric("torch_indices_", function(self) standardGeneric("torch_indices_"))
setGeneric("torch_values_", function(self) standardGeneric("torch_values_"))
setGeneric("torch_numel_", function(self) standardGeneric("torch_numel_"))
setGeneric("torch_unbind_", function(self, dim) standardGeneric("torch_unbind_"))
setGeneric("torch_to_sparse_", function(self, sparse_dim) standardGeneric("torch_to_sparse_"))
setGeneric("torch_to_mkldnn_", function(self) standardGeneric("torch_to_mkldnn_"))
setGeneric("torch_dequantize_", function(self) standardGeneric("torch_dequantize_"))
setGeneric("torch_q_scale_", function(self) standardGeneric("torch_q_scale_"))
setGeneric("torch_q_zero_point_", function(self) standardGeneric("torch_q_zero_point_"))
setGeneric("torch_int_repr_", function(self) standardGeneric("torch_int_repr_"))
setGeneric("torch_qscheme_", function(self) standardGeneric("torch_qscheme_"))
setGeneric("torch_to_", function(self, options, device, other, dtype, non_blocking, copy) standardGeneric("torch_to_"))
setGeneric("torch_item_", function(self) standardGeneric("torch_item_"))
setGeneric("torch_set__", function(self, source, storage_offset, size, stride) standardGeneric("torch_set__"))
setGeneric("torch_set_quantizer__", function(self, quantizer) standardGeneric("torch_set_quantizer__"))
setGeneric("torch_is_set_to_", function(self, tensor) standardGeneric("torch_is_set_to_"))
setGeneric("torch_masked_fill__", function(self, mask, value) standardGeneric("torch_masked_fill__"))
setGeneric("torch_masked_fill_", function(self, mask, value) standardGeneric("torch_masked_fill_"))
setGeneric("torch_masked_scatter__", function(self, mask, source) standardGeneric("torch_masked_scatter__"))
setGeneric("torch_masked_scatter_", function(self, mask, source) standardGeneric("torch_masked_scatter_"))
setGeneric("torch_view_", function(self, size) standardGeneric("torch_view_"))
setGeneric("torch_put__", function(self, index, source, accumulate) standardGeneric("torch_put__"))
setGeneric("torch_index_add__", function(self, dim, index, source) standardGeneric("torch_index_add__"))
setGeneric("torch_index_add_", function(self, dim, index, source) standardGeneric("torch_index_add_"))
setGeneric("torch_index_fill__", function(self, dim, index, value) standardGeneric("torch_index_fill__"))
setGeneric("torch_index_fill_", function(self, dim, index, value) standardGeneric("torch_index_fill_"))
setGeneric("torch_scatter__", function(self, dim, index, src, value) standardGeneric("torch_scatter__"))
setGeneric("torch_scatter_", function(self, dim, index, src, value) standardGeneric("torch_scatter_"))
setGeneric("torch_scatter_add__", function(self, dim, index, src) standardGeneric("torch_scatter_add__"))
setGeneric("torch_scatter_add_", function(self, dim, index, src) standardGeneric("torch_scatter_add_"))
setGeneric("torch_lt__", function(self, other) standardGeneric("torch_lt__"))
setGeneric("torch_gt__", function(self, other) standardGeneric("torch_gt__"))
setGeneric("torch_le__", function(self, other) standardGeneric("torch_le__"))
setGeneric("torch_ge__", function(self, other) standardGeneric("torch_ge__"))
setGeneric("torch_eq__", function(self, other) standardGeneric("torch_eq__"))
setGeneric("torch_ne__", function(self, other) standardGeneric("torch_ne__"))
setGeneric("torch___and___", function(self, other) standardGeneric("torch___and___"))
setGeneric("torch___iand___", function(self, other) standardGeneric("torch___iand___"))
setGeneric("torch___or___", function(self, other) standardGeneric("torch___or___"))
setGeneric("torch___ior___", function(self, other) standardGeneric("torch___ior___"))
setGeneric("torch___xor___", function(self, other) standardGeneric("torch___xor___"))
setGeneric("torch___ixor___", function(self, other) standardGeneric("torch___ixor___"))
setGeneric("torch___lshift___", function(self, other) standardGeneric("torch___lshift___"))
setGeneric("torch___ilshift___", function(self, other) standardGeneric("torch___ilshift___"))
setGeneric("torch___rshift___", function(self, other) standardGeneric("torch___rshift___"))
setGeneric("torch___irshift___", function(self, other) standardGeneric("torch___irshift___"))
setGeneric("torch_lgamma__", function(self) standardGeneric("torch_lgamma__"))
setGeneric("torch_atan2__", function(self, other) standardGeneric("torch_atan2__"))
setGeneric("torch_tril__", function(self, diagonal) standardGeneric("torch_tril__"))
setGeneric("torch_triu__", function(self, diagonal) standardGeneric("torch_triu__"))
setGeneric("torch_digamma__", function(self) standardGeneric("torch_digamma__"))
setGeneric("torch_polygamma__", function(self, n) standardGeneric("torch_polygamma__"))
setGeneric("torch_erfinv__", function(self) standardGeneric("torch_erfinv__"))
setGeneric("torch_renorm__", function(self, p, dim, maxnorm) standardGeneric("torch_renorm__"))
setGeneric("torch_pow__", function(self, exponent) standardGeneric("torch_pow__"))
setGeneric("torch_lerp__", function(self, end, weight) standardGeneric("torch_lerp__"))
setGeneric("torch_sign__", function(self) standardGeneric("torch_sign__"))
setGeneric("torch_fmod__", function(self, other) standardGeneric("torch_fmod__"))
setGeneric("torch_remainder__", function(self, other) standardGeneric("torch_remainder__"))
setGeneric("torch_addbmm__", function(self, batch1, batch2, beta, alpha) standardGeneric("torch_addbmm__"))
setGeneric("torch_addbmm_", function(self, batch1, batch2, beta, alpha) standardGeneric("torch_addbmm_"))
setGeneric("torch_addcmul__", function(self, tensor1, tensor2, value) standardGeneric("torch_addcmul__"))
setGeneric("torch_addcdiv__", function(self, tensor1, tensor2, value) standardGeneric("torch_addcdiv__"))
setGeneric("torch_random__", function(self, from, to) standardGeneric("torch_random__"))
setGeneric("torch_uniform__", function(self, from, to) standardGeneric("torch_uniform__"))
setGeneric("torch_normal__", function(self, mean, std) standardGeneric("torch_normal__"))
setGeneric("torch_cauchy__", function(self, median, sigma) standardGeneric("torch_cauchy__"))
setGeneric("torch_log_normal__", function(self, mean, std) standardGeneric("torch_log_normal__"))
setGeneric("torch_exponential__", function(self, lambd) standardGeneric("torch_exponential__"))
setGeneric("torch_geometric__", function(self, p) standardGeneric("torch_geometric__"))
setGeneric("torch_diag_", function(self, diagonal) standardGeneric("torch_diag_"))
setGeneric("torch_cross_", function(self, other, dim) standardGeneric("torch_cross_"))
setGeneric("torch_triu_", function(self, diagonal) standardGeneric("torch_triu_"))
setGeneric("torch_tril_", function(self, diagonal) standardGeneric("torch_tril_"))
setGeneric("torch_trace_", function(self) standardGeneric("torch_trace_"))
setGeneric("torch_ne_", function(self, other) standardGeneric("torch_ne_"))
setGeneric("torch_eq_", function(self, other) standardGeneric("torch_eq_"))
setGeneric("torch_ge_", function(self, other) standardGeneric("torch_ge_"))
setGeneric("torch_le_", function(self, other) standardGeneric("torch_le_"))
setGeneric("torch_gt_", function(self, other) standardGeneric("torch_gt_"))
setGeneric("torch_lt_", function(self, other) standardGeneric("torch_lt_"))
setGeneric("torch_take_", function(self, index) standardGeneric("torch_take_"))
setGeneric("torch_index_select_", function(self, dim, index) standardGeneric("torch_index_select_"))
setGeneric("torch_masked_select_", function(self, mask) standardGeneric("torch_masked_select_"))
setGeneric("torch_nonzero_", function(self) standardGeneric("torch_nonzero_"))
setGeneric("torch_nonzero_numpy_", function(self) standardGeneric("torch_nonzero_numpy_"))
setGeneric("torch_gather_", function(self, dim, index, sparse_grad) standardGeneric("torch_gather_"))
setGeneric("torch_addcmul_", function(self, tensor1, tensor2, value) standardGeneric("torch_addcmul_"))
setGeneric("torch_addcdiv_", function(self, tensor1, tensor2, value) standardGeneric("torch_addcdiv_"))
setGeneric("torch_lstsq_", function(self, A) standardGeneric("torch_lstsq_"))
setGeneric("torch_triangular_solve_", function(self, A, upper, transpose, unitriangular) standardGeneric("torch_triangular_solve_"))
setGeneric("torch_symeig_", function(self, eigenvectors, upper) standardGeneric("torch_symeig_"))
setGeneric("torch_eig_", function(self, eigenvectors) standardGeneric("torch_eig_"))
setGeneric("torch_svd_", function(self, some, compute_uv) standardGeneric("torch_svd_"))
setGeneric("torch_cholesky_", function(self, upper) standardGeneric("torch_cholesky_"))
setGeneric("torch_cholesky_solve_", function(self, input2, upper) standardGeneric("torch_cholesky_solve_"))
setGeneric("torch_solve_", function(self, A) standardGeneric("torch_solve_"))
setGeneric("torch_cholesky_inverse_", function(self, upper) standardGeneric("torch_cholesky_inverse_"))
setGeneric("torch_qr_", function(self, some) standardGeneric("torch_qr_"))
setGeneric("torch_geqrf_", function(self) standardGeneric("torch_geqrf_"))
setGeneric("torch_orgqr_", function(self, input2) standardGeneric("torch_orgqr_"))
setGeneric("torch_ormqr_", function(self, input2, input3, left, transpose) standardGeneric("torch_ormqr_"))
setGeneric("torch_lu_solve_", function(self, LU_data, LU_pivots) standardGeneric("torch_lu_solve_"))
setGeneric("torch_multinomial_", function(self, num_samples, replacement) standardGeneric("torch_multinomial_"))
setGeneric("torch_lgamma_", function(self) standardGeneric("torch_lgamma_"))
setGeneric("torch_digamma_", function(self) standardGeneric("torch_digamma_"))
setGeneric("torch_polygamma_", function(n, self) standardGeneric("torch_polygamma_"))
setGeneric("torch_erfinv_", function(self) standardGeneric("torch_erfinv_"))
setGeneric("torch_dist_", function(self, other, p) standardGeneric("torch_dist_"))
setGeneric("torch_atan2_", function(self, other) standardGeneric("torch_atan2_"))
setGeneric("torch_lerp_", function(self, end, weight) standardGeneric("torch_lerp_"))
setGeneric("torch_histc_", function(self, bins, min, max) standardGeneric("torch_histc_"))
setGeneric("torch_sign_", function(self) standardGeneric("torch_sign_"))
setGeneric("torch_fmod_", function(self, other) standardGeneric("torch_fmod_"))
setGeneric("torch_remainder_", function(self, other) standardGeneric("torch_remainder_"))
setGeneric("torch_sort_", function(self, dim, descending) standardGeneric("torch_sort_"))
setGeneric("torch_argsort_", function(self, dim, descending) standardGeneric("torch_argsort_"))
setGeneric("torch_topk_", function(self, k, dim, largest, sorted) standardGeneric("torch_topk_"))
setGeneric("torch_renorm_", function(self, p, dim, maxnorm) standardGeneric("torch_renorm_"))
setGeneric("torch_unfold_", function(self, dimension, size, step) standardGeneric("torch_unfold_"))
setGeneric("torch_equal_", function(self, other) standardGeneric("torch_equal_"))
setGeneric("torch_alias_", function(self) standardGeneric("torch_alias_"))


setMethod(
 f='torch_backward_',
 signature=list(self='externalptr', gradient='externalptr', keep_graph='logical', create_graph='logical'),
 definition=function(self, gradient, keep_graph, create_graph) {
torch_backward_0427181972d30e1747ec208d30a7470a(self, gradient, keep_graph, create_graph)
 }
)
setMethod(
 f='torch_set_data_',
 signature=list(self='externalptr', new_data='externalptr'),
 definition=function(self, new_data) {
torch_set_data_89728a9882441111256f356aa2c7bd2d(self, new_data)
 }
)
setMethod(
 f='torch_abs_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_abs_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_abs__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_abs__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_acos_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_acos_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_acos__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_acos__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_add_',
 signature=list(self='externalptr', other='externalptr', alpha='numeric'),
 definition=function(self, other, alpha) {
torch_add_02bec2e8e54b6237090a5042dd1e991c(self, other, alpha)
 }
)
setMethod(
 f='torch_add__',
 signature=list(self='externalptr', other='externalptr', alpha='numeric'),
 definition=function(self, other, alpha) {
torch_add__6b3a76c4ed9f62ef65e1cec9d661dfa5(self, other, alpha)
 }
)
setMethod(
 f='torch_add_',
 signature=list(self='externalptr', other='numeric', alpha='numeric'),
 definition=function(self, other, alpha) {
torch_add_783c8a98771291069457898dc444b949(self, other, alpha)
 }
)
setMethod(
 f='torch_add__',
 signature=list(self='externalptr', other='numeric', alpha='numeric'),
 definition=function(self, other, alpha) {
torch_add__407c973b091671443effd1a0b4dfe28f(self, other, alpha)
 }
)
setMethod(
 f='torch_addmv_',
 signature=list(self='externalptr', mat='externalptr', vec='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, mat, vec, beta, alpha) {
torch_addmv_1072d7a4d7a0dab6482cb299987e26ff(self, mat, vec, beta, alpha)
 }
)
setMethod(
 f='torch_addmv__',
 signature=list(self='externalptr', mat='externalptr', vec='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, mat, vec, beta, alpha) {
torch_addmv__4820c49a5d1d67877f59e1d57a262a9d(self, mat, vec, beta, alpha)
 }
)
setMethod(
 f='torch_addr_',
 signature=list(self='externalptr', vec1='externalptr', vec2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, vec1, vec2, beta, alpha) {
torch_addr_ab5fbf3dbf88b209e751ed0770b30e5c(self, vec1, vec2, beta, alpha)
 }
)
setMethod(
 f='torch_addr__',
 signature=list(self='externalptr', vec1='externalptr', vec2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, vec1, vec2, beta, alpha) {
torch_addr__d04ad2d4b220f065f90babfb039cafc0(self, vec1, vec2, beta, alpha)
 }
)
setMethod(
 f='torch_all_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_all_a00d65de0c17931eb6106e885279f146(self, dim, keepdim)
 }
)
setMethod(
 f='torch_allclose_',
 signature=list(self='externalptr', other='externalptr', rtol='double', atol='double', equal_nan='logical'),
 definition=function(self, other, rtol, atol, equal_nan) {
torch_allclose_68991204bc2d1d5ac874203da18b195a(self, other, rtol, atol, equal_nan)
 }
)
setMethod(
 f='torch_any_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_any_a00d65de0c17931eb6106e885279f146(self, dim, keepdim)
 }
)
setMethod(
 f='torch_argmax_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_argmax_49e69c8c7172be362fffe33e9ea73ccb(self, dim, keepdim)
 }
)
setMethod(
 f='torch_argmin_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_argmin_49e69c8c7172be362fffe33e9ea73ccb(self, dim, keepdim)
 }
)
setMethod(
 f='torch_as_strided_',
 signature=list(self='externalptr', size='numeric', stride='numeric', storage_offset='numeric'),
 definition=function(self, size, stride, storage_offset) {
torch_as_strided_48a5ca9ebdf2e93013f04622e447af23(self, size, stride, storage_offset)
 }
)
setMethod(
 f='torch_as_strided__',
 signature=list(self='externalptr', size='numeric', stride='numeric', storage_offset='numeric'),
 definition=function(self, size, stride, storage_offset) {
torch_as_strided__854a1e8405883f30f2ac785b62b9df35(self, size, stride, storage_offset)
 }
)
setMethod(
 f='torch_asin_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_asin_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_asin__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_asin__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_atan_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_atan_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_atan__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_atan__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_baddbmm_',
 signature=list(self='externalptr', batch1='externalptr', batch2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, batch1, batch2, beta, alpha) {
torch_baddbmm_f084ab37fd7cee9fb89f911187c51117(self, batch1, batch2, beta, alpha)
 }
)
setMethod(
 f='torch_baddbmm__',
 signature=list(self='externalptr', batch1='externalptr', batch2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, batch1, batch2, beta, alpha) {
torch_baddbmm__e39679a201c809cf04d4768abd5f9472(self, batch1, batch2, beta, alpha)
 }
)
setMethod(
 f='torch_bernoulli_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_bernoulli_172e9e81db927896b92b352cb077113b(self)
 }
)
setMethod(
 f='torch_bernoulli__',
 signature=list(self='externalptr', p='externalptr'),
 definition=function(self, p) {
torch_bernoulli__a2c4898743bb0b00055eccf935534a10(self, p)
 }
)
setMethod(
 f='torch_bernoulli__',
 signature=list(self='externalptr', p='double'),
 definition=function(self, p) {
torch_bernoulli__c0d7479cf591e60afc5ecc0adace5091(self, p)
 }
)
setMethod(
 f='torch_bernoulli_',
 signature=list(self='externalptr', p='double'),
 definition=function(self, p) {
torch_bernoulli_7c6a3063d6c9160addd43f700047fc1d(self, p)
 }
)
setMethod(
 f='torch_bincount_',
 signature=list(self='externalptr', weights='externalptr', minlength='numeric'),
 definition=function(self, weights, minlength) {
torch_bincount_f4aaee0119fb1bbeda5d6924832e6ccd(self, weights, minlength)
 }
)
setMethod(
 f='torch_bitwise_not_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_bitwise_not_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_bitwise_not__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_bitwise_not__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_bmm_',
 signature=list(self='externalptr', mat2='externalptr'),
 definition=function(self, mat2) {
torch_bmm_4179872982cdd9692065c1d0412bcd54(self, mat2)
 }
)
setMethod(
 f='torch_ceil_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_ceil_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_ceil__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_ceil__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_chunk_',
 signature=list(self='externalptr', chunks='numeric', dim='numeric'),
 definition=function(self, chunks, dim) {
torch_chunk_d1ba7cb9ab45bed064482d6655a047ba(self, chunks, dim)
 }
)
setMethod(
 f='torch_clamp_',
 signature=list(self='externalptr', min='numeric', max='numeric'),
 definition=function(self, min, max) {
torch_clamp_70f48b9c0a72bac849220903ec22a50a(self, min, max)
 }
)
setMethod(
 f='torch_clamp__',
 signature=list(self='externalptr', min='numeric', max='numeric'),
 definition=function(self, min, max) {
torch_clamp__de153b8f9dac7cff075500db6ee1f472(self, min, max)
 }
)
setMethod(
 f='torch_clamp_max_',
 signature=list(self='externalptr', max='numeric'),
 definition=function(self, max) {
torch_clamp_max_632290f0e396f400e950acd849169431(self, max)
 }
)
setMethod(
 f='torch_clamp_max__',
 signature=list(self='externalptr', max='numeric'),
 definition=function(self, max) {
torch_clamp_max__1319c907c5eeba718b7832b2e9395f61(self, max)
 }
)
setMethod(
 f='torch_clamp_min_',
 signature=list(self='externalptr', min='numeric'),
 definition=function(self, min) {
torch_clamp_min_96a46b6306a546d929ea4f53bd544b90(self, min)
 }
)
setMethod(
 f='torch_clamp_min__',
 signature=list(self='externalptr', min='numeric'),
 definition=function(self, min) {
torch_clamp_min__a683f5abd859c5ad58289c38de974579(self, min)
 }
)
setMethod(
 f='torch_contiguous_',
 signature=list(self='externalptr', memory_format='character'),
 definition=function(self, memory_format) {
torch_contiguous_d11f20d0c3dedbed7060d922567eeb82(self, memory_format)
 }
)
setMethod(
 f='torch_copy__',
 signature=list(self='externalptr', src='externalptr', non_blocking='logical'),
 definition=function(self, src, non_blocking) {
torch_copy__1ccfd2368a7db3f7bcf684e5471b9cbe(self, src, non_blocking)
 }
)
setMethod(
 f='torch_cos_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_cos_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_cos__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_cos__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_cosh_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_cosh_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_cosh__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_cosh__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_cumsum_',
 signature=list(self='externalptr', dim='numeric', dtype='character'),
 definition=function(self, dim, dtype) {
torch_cumsum_497aca3433c567f5542cfedd28714419(self, dim, dtype)
 }
)
setMethod(
 f='torch_cumprod_',
 signature=list(self='externalptr', dim='numeric', dtype='character'),
 definition=function(self, dim, dtype) {
torch_cumprod_497aca3433c567f5542cfedd28714419(self, dim, dtype)
 }
)
setMethod(
 f='torch_det_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_det_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_diag_embed_',
 signature=list(self='externalptr', offset='numeric', dim1='numeric', dim2='numeric'),
 definition=function(self, offset, dim1, dim2) {
torch_diag_embed_83af1f3094a6ddea435a5fde5762756f(self, offset, dim1, dim2)
 }
)
setMethod(
 f='torch_diagflat_',
 signature=list(self='externalptr', offset='numeric'),
 definition=function(self, offset) {
torch_diagflat_c2bc38d3bfe8c6855ff8eac707000e71(self, offset)
 }
)
setMethod(
 f='torch_diagonal_',
 signature=list(self='externalptr', offset='numeric', dim1='numeric', dim2='numeric'),
 definition=function(self, offset, dim1, dim2) {
torch_diagonal_83af1f3094a6ddea435a5fde5762756f(self, offset, dim1, dim2)
 }
)
setMethod(
 f='torch_fill_diagonal__',
 signature=list(self='externalptr', fill_value='numeric', wrap='logical'),
 definition=function(self, fill_value, wrap) {
torch_fill_diagonal__afd80df47ca4531a8d0038af1e0e0e29(self, fill_value, wrap)
 }
)
setMethod(
 f='torch_div_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_div_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_div__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_div__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_div_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_div_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_div__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_div__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_dot_',
 signature=list(self='externalptr', tensor='externalptr'),
 definition=function(self, tensor) {
torch_dot_20e8a5d03011737e0a350fd8208fc32a(self, tensor)
 }
)
setMethod(
 f='torch_resize__',
 signature=list(self='externalptr', size='numeric'),
 definition=function(self, size) {
torch_resize__b5c406f407e1edb269082d30571274a2(self, size)
 }
)
setMethod(
 f='torch_erf_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_erf_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_erf__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_erf__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_erfc_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_erfc_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_erfc__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_erfc__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_exp_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_exp_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_exp__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_exp__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_expm1_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_expm1_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_expm1__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_expm1__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_expand_',
 signature=list(self='externalptr', size='numeric', implicit='logical'),
 definition=function(self, size, implicit) {
torch_expand_8fb715dc6dd8860dc19da19157c6a569(self, size, implicit)
 }
)
setMethod(
 f='torch_expand_as_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_expand_as_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_flatten_',
 signature=list(self='externalptr', start_dim='numeric', end_dim='numeric'),
 definition=function(self, start_dim, end_dim) {
torch_flatten_81cfa679173f7cf487978866e858e0d7(self, start_dim, end_dim)
 }
)
setMethod(
 f='torch_fill__',
 signature=list(self='externalptr', value='numeric'),
 definition=function(self, value) {
torch_fill__07130a53d127abf9a9f43e9ab5623ac4(self, value)
 }
)
setMethod(
 f='torch_fill__',
 signature=list(self='externalptr', value='externalptr'),
 definition=function(self, value) {
torch_fill__e477b33acb65374ba9ec28a77721a45e(self, value)
 }
)
setMethod(
 f='torch_floor_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_floor_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_floor__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_floor__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_frac_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_frac_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_frac__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_frac__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_ger_',
 signature=list(self='externalptr', vec2='externalptr'),
 definition=function(self, vec2) {
torch_ger_5c12932d2aad63902edd715966d08934(self, vec2)
 }
)
setMethod(
 f='torch_fft_',
 signature=list(self='externalptr', signal_ndim='numeric', normalized='logical'),
 definition=function(self, signal_ndim, normalized) {
torch_fft_b40ab26de564940dc199a685b56a303c(self, signal_ndim, normalized)
 }
)
setMethod(
 f='torch_ifft_',
 signature=list(self='externalptr', signal_ndim='numeric', normalized='logical'),
 definition=function(self, signal_ndim, normalized) {
torch_ifft_b40ab26de564940dc199a685b56a303c(self, signal_ndim, normalized)
 }
)
setMethod(
 f='torch_rfft_',
 signature=list(self='externalptr', signal_ndim='numeric', normalized='logical', onesided='logical'),
 definition=function(self, signal_ndim, normalized, onesided) {
torch_rfft_952cd2ec36e935286dd362c64eddbc76(self, signal_ndim, normalized, onesided)
 }
)
setMethod(
 f='torch_irfft_',
 signature=list(self='externalptr', signal_ndim='numeric', normalized='logical', onesided='logical', signal_sizes='numeric'),
 definition=function(self, signal_ndim, normalized, onesided, signal_sizes) {
torch_irfft_8dbe0873d7287e95651dac3536aaaab4(self, signal_ndim, normalized, onesided, signal_sizes)
 }
)
setMethod(
 f='torch_index_',
 signature=list(self='externalptr', indices='list'),
 definition=function(self, indices) {
torch_index_814b37e08ed831a0d37f0f073ffe1b56(self, indices)
 }
)
setMethod(
 f='torch_index_copy__',
 signature=list(self='externalptr', dim='numeric', index='externalptr', source='externalptr'),
 definition=function(self, dim, index, source) {
torch_index_copy__0be4630fe46f5a3434059ed7debb5603(self, dim, index, source)
 }
)
setMethod(
 f='torch_index_copy_',
 signature=list(self='externalptr', dim='numeric', index='externalptr', source='externalptr'),
 definition=function(self, dim, index, source) {
torch_index_copy_073fd53276f8ca2975951ef418c77979(self, dim, index, source)
 }
)
setMethod(
 f='torch_index_put__',
 signature=list(self='externalptr', indices='list', values='externalptr', accumulate='logical'),
 definition=function(self, indices, values, accumulate) {
torch_index_put__59690c3dcdbff4a75d67259ec5ecedd7(self, indices, values, accumulate)
 }
)
setMethod(
 f='torch_index_put_',
 signature=list(self='externalptr', indices='list', values='externalptr', accumulate='logical'),
 definition=function(self, indices, values, accumulate) {
torch_index_put_17d445458eace87a93650394814e7abd(self, indices, values, accumulate)
 }
)
setMethod(
 f='torch_inverse_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_inverse_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_isclose_',
 signature=list(self='externalptr', other='externalptr', rtol='double', atol='double', equal_nan='logical'),
 definition=function(self, other, rtol, atol, equal_nan) {
torch_isclose_68991204bc2d1d5ac874203da18b195a(self, other, rtol, atol, equal_nan)
 }
)
setMethod(
 f='torch_is_distributed_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_is_distributed_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_is_floating_point_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_is_floating_point_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_is_complex_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_is_complex_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_is_nonzero_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_is_nonzero_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_is_same_size_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_is_same_size_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_is_signed_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_is_signed_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_kthvalue_',
 signature=list(self='externalptr', k='numeric', dim='numeric', keepdim='logical'),
 definition=function(self, k, dim, keepdim) {
torch_kthvalue_552a2f1f5a868450b9902d289f1bc34a(self, k, dim, keepdim)
 }
)
setMethod(
 f='torch_log_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_log_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_log__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_log__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_log10_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_log10_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_log10__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_log10__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_log1p_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_log1p_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_log1p__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_log1p__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_log2_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_log2_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_log2__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_log2__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_logdet_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_logdet_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_log_softmax_',
 signature=list(self='externalptr', dim='numeric', dtype='character'),
 definition=function(self, dim, dtype) {
torch_log_softmax_497aca3433c567f5542cfedd28714419(self, dim, dtype)
 }
)
setMethod(
 f='torch_logsumexp_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_logsumexp_3d8e8f11c9689ef6a75a9c7ca8d6b7b6(self, dim, keepdim)
 }
)
setMethod(
 f='torch_matmul_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_matmul_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_matrix_power_',
 signature=list(self='externalptr', n='numeric'),
 definition=function(self, n) {
torch_matrix_power_fbba1f43b92fd902db09be10350c6bdf(self, n)
 }
)
setMethod(
 f='torch_max_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_max_a00d65de0c17931eb6106e885279f146(self, dim, keepdim)
 }
)
setMethod(
 f='torch_max_values_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_max_values_3d8e8f11c9689ef6a75a9c7ca8d6b7b6(self, dim, keepdim)
 }
)
setMethod(
 f='torch_mean_',
 signature=list(self='externalptr', dtype='character'),
 definition=function(self, dtype) {
torch_mean_e9eab74b972ba6cab392179a4f0a1630(self, dtype)
 }
)
setMethod(
 f='torch_mean_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical', dtype='character'),
 definition=function(self, dim, keepdim, dtype) {
torch_mean_b75b9fc8150d6ec976bec5e79e01d1d3(self, dim, keepdim, dtype)
 }
)
setMethod(
 f='torch_median_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_median_a00d65de0c17931eb6106e885279f146(self, dim, keepdim)
 }
)
setMethod(
 f='torch_min_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_min_a00d65de0c17931eb6106e885279f146(self, dim, keepdim)
 }
)
setMethod(
 f='torch_min_values_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_min_values_3d8e8f11c9689ef6a75a9c7ca8d6b7b6(self, dim, keepdim)
 }
)
setMethod(
 f='torch_mm_',
 signature=list(self='externalptr', mat2='externalptr'),
 definition=function(self, mat2) {
torch_mm_4179872982cdd9692065c1d0412bcd54(self, mat2)
 }
)
setMethod(
 f='torch_mode_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical'),
 definition=function(self, dim, keepdim) {
torch_mode_a00d65de0c17931eb6106e885279f146(self, dim, keepdim)
 }
)
setMethod(
 f='torch_mul_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_mul_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_mul__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_mul__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_mul_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_mul_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_mul__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_mul__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_mv_',
 signature=list(self='externalptr', vec='externalptr'),
 definition=function(self, vec) {
torch_mv_98e027563655aaef18dcf0fdd62aa4e7(self, vec)
 }
)
setMethod(
 f='torch_mvlgamma_',
 signature=list(self='externalptr', p='numeric'),
 definition=function(self, p) {
torch_mvlgamma_9a31517035e3d70163aa3a03599742d1(self, p)
 }
)
setMethod(
 f='torch_mvlgamma__',
 signature=list(self='externalptr', p='numeric'),
 definition=function(self, p) {
torch_mvlgamma__0c99e2d4b756f575ccf8a595a519a030(self, p)
 }
)
setMethod(
 f='torch_narrow_copy_',
 signature=list(self='externalptr', dim='numeric', start='numeric', length='numeric'),
 definition=function(self, dim, start, length) {
torch_narrow_copy_d65beda5fa619a4b42b0a4f5423ae7af(self, dim, start, length)
 }
)
setMethod(
 f='torch_narrow_',
 signature=list(self='externalptr', dim='numeric', start='numeric', length='numeric'),
 definition=function(self, dim, start, length) {
torch_narrow_d65beda5fa619a4b42b0a4f5423ae7af(self, dim, start, length)
 }
)
setMethod(
 f='torch_permute_',
 signature=list(self='externalptr', dims='numeric'),
 definition=function(self, dims) {
torch_permute_58ccf3f283122d340a673beb6bd417b2(self, dims)
 }
)
setMethod(
 f='torch_numpy_T_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_numpy_T_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_pin_memory_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_pin_memory_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_pinverse_',
 signature=list(self='externalptr', rcond='double'),
 definition=function(self, rcond) {
torch_pinverse_9451f1dc9df4ad12effb96865b7d5cc1(self, rcond)
 }
)
setMethod(
 f='torch_reciprocal_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_reciprocal_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_reciprocal__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_reciprocal__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_neg_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_neg_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_neg__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_neg__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_repeat_',
 signature=list(self='externalptr', repeats='numeric'),
 definition=function(self, repeats) {
torch_repeat_eefba09ed344a4842dade390b5fd0e82(self, repeats)
 }
)
setMethod(
 f='torch_repeat_interleave_',
 signature=list(self='externalptr', repeats='externalptr', dim='numeric'),
 definition=function(self, repeats, dim) {
torch_repeat_interleave_df8da4cd3c0efd9bcbd034f12cd6938f(self, repeats, dim)
 }
)
setMethod(
 f='torch_repeat_interleave_',
 signature=list(self='externalptr', repeats='numeric', dim='numeric'),
 definition=function(self, repeats, dim) {
torch_repeat_interleave_f4cdbd612fc56bd598939b1ed8c2ac87(self, repeats, dim)
 }
)
setMethod(
 f='torch_reshape_',
 signature=list(self='externalptr', shape='numeric'),
 definition=function(self, shape) {
torch_reshape_00ebe0c7086ba70f14f870d6a23ec6dd(self, shape)
 }
)
setMethod(
 f='torch_reshape_as_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_reshape_as_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_round_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_round_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_round__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_round__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_relu_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_relu_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_relu__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_relu__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_prelu_',
 signature=list(self='externalptr', weight='externalptr'),
 definition=function(self, weight) {
torch_prelu_74aa237ed2400169d048df729a1d6f92(self, weight)
 }
)
setMethod(
 f='torch_prelu_backward_',
 signature=list(grad_output='externalptr', self='externalptr', weight='externalptr'),
 definition=function(grad_output, self, weight) {
torch_prelu_backward_fdadff1d8b3c27a6244816a269247822(grad_output, self, weight)
 }
)
setMethod(
 f='torch_hardshrink_',
 signature=list(self='externalptr', lambd='numeric'),
 definition=function(self, lambd) {
torch_hardshrink_899d60f0bf515c42a60b3525f217fd29(self, lambd)
 }
)
setMethod(
 f='torch_hardshrink_backward_',
 signature=list(grad_out='externalptr', self='externalptr', lambd='numeric'),
 definition=function(grad_out, self, lambd) {
torch_hardshrink_backward_ae549be1d694fc273fb6a873601ddca5(grad_out, self, lambd)
 }
)
setMethod(
 f='torch_rsqrt_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_rsqrt_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_rsqrt__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_rsqrt__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_select_',
 signature=list(self='externalptr', dim='numeric', index='numeric'),
 definition=function(self, dim, index) {
torch_select_b059a0edb6085ce5612a62f96a05b06f(self, dim, index)
 }
)
setMethod(
 f='torch_sigmoid_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sigmoid_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_sigmoid__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sigmoid__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_sin_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sin_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_sin__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sin__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_sinh_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sinh_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_sinh__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sinh__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_detach_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_detach_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_detach__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_detach__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_size_',
 signature=list(self='externalptr', dim='numeric'),
 definition=function(self, dim) {
torch_size_ec971b768b5201c8c4ad6177a2be95ab(self, dim)
 }
)
setMethod(
 f='torch_slice_',
 signature=list(self='externalptr', dim='numeric', start='numeric', end='numeric', step='numeric'),
 definition=function(self, dim, start, end, step) {
torch_slice_43b6737dc8edf11756418078b04e3d84(self, dim, start, end, step)
 }
)
setMethod(
 f='torch_slogdet_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_slogdet_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_smm_',
 signature=list(self='externalptr', mat2='externalptr'),
 definition=function(self, mat2) {
torch_smm_4179872982cdd9692065c1d0412bcd54(self, mat2)
 }
)
setMethod(
 f='torch_softmax_',
 signature=list(self='externalptr', dim='numeric', dtype='character'),
 definition=function(self, dim, dtype) {
torch_softmax_497aca3433c567f5542cfedd28714419(self, dim, dtype)
 }
)
setMethod(
 f='torch_split_',
 signature=list(self='externalptr', split_size='numeric', dim='numeric'),
 definition=function(self, split_size, dim) {
torch_split_899f07e47445429ec4e214929e1deb92(self, split_size, dim)
 }
)
setMethod(
 f='torch_split_with_sizes_',
 signature=list(self='externalptr', split_sizes='numeric', dim='numeric'),
 definition=function(self, split_sizes, dim) {
torch_split_with_sizes_6a9223a9c3eb63442d2bc65531b2f0cf(self, split_sizes, dim)
 }
)
setMethod(
 f='torch_squeeze_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_squeeze_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_squeeze_',
 signature=list(self='externalptr', dim='numeric'),
 definition=function(self, dim) {
torch_squeeze_ec971b768b5201c8c4ad6177a2be95ab(self, dim)
 }
)
setMethod(
 f='torch_squeeze__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_squeeze__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_squeeze__',
 signature=list(self='externalptr', dim='numeric'),
 definition=function(self, dim) {
torch_squeeze__db1e43f53c3b816eb931c1c0de0f3b94(self, dim)
 }
)
setMethod(
 f='torch_sspaddmm_',
 signature=list(self='externalptr', mat1='externalptr', mat2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, mat1, mat2, beta, alpha) {
torch_sspaddmm_593f90a5e31a632afec95421dabb830b(self, mat1, mat2, beta, alpha)
 }
)
setMethod(
 f='torch_stft_',
 signature=list(self='externalptr', n_fft='numeric', hop_length='numeric', win_length='numeric', window='externalptr', normalized='logical', onesided='logical'),
 definition=function(self, n_fft, hop_length, win_length, window, normalized, onesided) {
torch_stft_c07c54f0f0d63b84174cb8de9bcc4ff5(self, n_fft, hop_length, win_length, window, normalized, onesided)
 }
)
setMethod(
 f='torch_stride_',
 signature=list(self='externalptr', dim='numeric'),
 definition=function(self, dim) {
torch_stride_ec971b768b5201c8c4ad6177a2be95ab(self, dim)
 }
)
setMethod(
 f='torch_sum_',
 signature=list(self='externalptr', dtype='character'),
 definition=function(self, dtype) {
torch_sum_e9eab74b972ba6cab392179a4f0a1630(self, dtype)
 }
)
setMethod(
 f='torch_sum_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical', dtype='character'),
 definition=function(self, dim, keepdim, dtype) {
torch_sum_b75b9fc8150d6ec976bec5e79e01d1d3(self, dim, keepdim, dtype)
 }
)
setMethod(
 f='torch_sum_to_size_',
 signature=list(self='externalptr', size='numeric'),
 definition=function(self, size) {
torch_sum_to_size_6bf55218cd71b219c054293f38520051(self, size)
 }
)
setMethod(
 f='torch_sqrt_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sqrt_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_sqrt__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sqrt__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_std_',
 signature=list(self='externalptr', unbiased='logical'),
 definition=function(self, unbiased) {
torch_std_268623e9239aead566394fd32dccb710(self, unbiased)
 }
)
setMethod(
 f='torch_std_',
 signature=list(self='externalptr', dim='numeric', unbiased='logical', keepdim='logical'),
 definition=function(self, dim, unbiased, keepdim) {
torch_std_e89c4c516b68404e86cc256eb5434c02(self, dim, unbiased, keepdim)
 }
)
setMethod(
 f='torch_prod_',
 signature=list(self='externalptr', dtype='character'),
 definition=function(self, dtype) {
torch_prod_e9eab74b972ba6cab392179a4f0a1630(self, dtype)
 }
)
setMethod(
 f='torch_prod_',
 signature=list(self='externalptr', dim='numeric', keepdim='logical', dtype='character'),
 definition=function(self, dim, keepdim, dtype) {
torch_prod_23927ad81b9cf9f7d2632329a9328fb7(self, dim, keepdim, dtype)
 }
)
setMethod(
 f='torch_t_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_t_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_t__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_t__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_tan_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_tan_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_tan__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_tan__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_tanh_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_tanh_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_tanh__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_tanh__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_transpose_',
 signature=list(self='externalptr', dim0='numeric', dim1='numeric'),
 definition=function(self, dim0, dim1) {
torch_transpose_79f15b60fe51bb82eeccf0ccc4b72c75(self, dim0, dim1)
 }
)
setMethod(
 f='torch_transpose__',
 signature=list(self='externalptr', dim0='numeric', dim1='numeric'),
 definition=function(self, dim0, dim1) {
torch_transpose__9862ce3a20e47cba29727a33f2297bdc(self, dim0, dim1)
 }
)
setMethod(
 f='torch_flip_',
 signature=list(self='externalptr', dims='numeric'),
 definition=function(self, dims) {
torch_flip_58ccf3f283122d340a673beb6bd417b2(self, dims)
 }
)
setMethod(
 f='torch_roll_',
 signature=list(self='externalptr', shifts='numeric', dims='numeric'),
 definition=function(self, shifts, dims) {
torch_roll_7d21b5356c80bf75ee8b086dbd93db76(self, shifts, dims)
 }
)
setMethod(
 f='torch_rot90_',
 signature=list(self='externalptr', k='numeric', dims='numeric'),
 definition=function(self, k, dims) {
torch_rot90_ba5ca2b0453143490cbeb12ac67a58eb(self, k, dims)
 }
)
setMethod(
 f='torch_trunc_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_trunc_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_trunc__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_trunc__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_type_as_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_type_as_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_unsqueeze_',
 signature=list(self='externalptr', dim='numeric'),
 definition=function(self, dim) {
torch_unsqueeze_ec971b768b5201c8c4ad6177a2be95ab(self, dim)
 }
)
setMethod(
 f='torch_unsqueeze__',
 signature=list(self='externalptr', dim='numeric'),
 definition=function(self, dim) {
torch_unsqueeze__db1e43f53c3b816eb931c1c0de0f3b94(self, dim)
 }
)
setMethod(
 f='torch_var_',
 signature=list(self='externalptr', unbiased='logical'),
 definition=function(self, unbiased) {
torch_var_268623e9239aead566394fd32dccb710(self, unbiased)
 }
)
setMethod(
 f='torch_var_',
 signature=list(self='externalptr', dim='numeric', unbiased='logical', keepdim='logical'),
 definition=function(self, dim, unbiased, keepdim) {
torch_var_e89c4c516b68404e86cc256eb5434c02(self, dim, unbiased, keepdim)
 }
)
setMethod(
 f='torch_view_as_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_view_as_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_where_',
 signature=list(condition='externalptr', self='externalptr', other='externalptr'),
 definition=function(condition, self, other) {
torch_where_659bd988fbd40cad9b2bc717050fdaa2(condition, self, other)
 }
)
setMethod(
 f='torch_norm_',
 signature=list(self='externalptr', p='numeric', dtype='character'),
 definition=function(self, p, dtype) {
torch_norm_030beec94146c3ad558fea0a5a1156c6(self, p, dtype)
 }
)
setMethod(
 f='torch_norm_',
 signature=list(self='externalptr', p='numeric'),
 definition=function(self, p) {
torch_norm_04a03af9c23e2262c0c6aca290ab7d4d(self, p)
 }
)
setMethod(
 f='torch_norm_',
 signature=list(self='externalptr', p='numeric', dim='numeric', keepdim='logical', dtype='character'),
 definition=function(self, p, dim, keepdim, dtype) {
torch_norm_2dfa12771319ec3116f4da72cb12b3d0(self, p, dim, keepdim, dtype)
 }
)
setMethod(
 f='torch_norm_',
 signature=list(self='externalptr', p='numeric', dim='numeric', keepdim='logical'),
 definition=function(self, p, dim, keepdim) {
torch_norm_6d38f8cf7270821224b49b321718823e(self, p, dim, keepdim)
 }
)
setMethod(
 f='torch_clone_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_clone_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_resize_as__',
 signature=list(self='externalptr', the_template='externalptr'),
 definition=function(self, the_template) {
torch_resize_as__1e3e95d16da5686b2ed8669167e56ef0(self, the_template)
 }
)
setMethod(
 f='torch_pow_',
 signature=list(self='externalptr', exponent='numeric'),
 definition=function(self, exponent) {
torch_pow_a504ebb51e840281521576607ce458c3(self, exponent)
 }
)
setMethod(
 f='torch_zero__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_zero__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_sub_',
 signature=list(self='externalptr', other='externalptr', alpha='numeric'),
 definition=function(self, other, alpha) {
torch_sub_02bec2e8e54b6237090a5042dd1e991c(self, other, alpha)
 }
)
setMethod(
 f='torch_sub__',
 signature=list(self='externalptr', other='externalptr', alpha='numeric'),
 definition=function(self, other, alpha) {
torch_sub__6b3a76c4ed9f62ef65e1cec9d661dfa5(self, other, alpha)
 }
)
setMethod(
 f='torch_sub_',
 signature=list(self='externalptr', other='numeric', alpha='numeric'),
 definition=function(self, other, alpha) {
torch_sub_783c8a98771291069457898dc444b949(self, other, alpha)
 }
)
setMethod(
 f='torch_sub__',
 signature=list(self='externalptr', other='numeric', alpha='numeric'),
 definition=function(self, other, alpha) {
torch_sub__407c973b091671443effd1a0b4dfe28f(self, other, alpha)
 }
)
setMethod(
 f='torch_addmm_',
 signature=list(self='externalptr', mat1='externalptr', mat2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, mat1, mat2, beta, alpha) {
torch_addmm_593f90a5e31a632afec95421dabb830b(self, mat1, mat2, beta, alpha)
 }
)
setMethod(
 f='torch_addmm__',
 signature=list(self='externalptr', mat1='externalptr', mat2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, mat1, mat2, beta, alpha) {
torch_addmm__e6db0e26c4b875bde9c015e72499386d(self, mat1, mat2, beta, alpha)
 }
)
setMethod(
 f='torch_sparse_resize__',
 signature=list(self='externalptr', size='numeric', sparse_dim='numeric', dense_dim='numeric'),
 definition=function(self, size, sparse_dim, dense_dim) {
torch_sparse_resize__f9b8d88bc0baafd239d11f2fcea30be5(self, size, sparse_dim, dense_dim)
 }
)
setMethod(
 f='torch_sparse_resize_and_clear__',
 signature=list(self='externalptr', size='numeric', sparse_dim='numeric', dense_dim='numeric'),
 definition=function(self, size, sparse_dim, dense_dim) {
torch_sparse_resize_and_clear__f9b8d88bc0baafd239d11f2fcea30be5(self, size, sparse_dim, dense_dim)
 }
)
setMethod(
 f='torch_sparse_mask_',
 signature=list(self='externalptr', mask='externalptr'),
 definition=function(self, mask) {
torch_sparse_mask_ed5bb55732ad3303430891d7fa131c57(self, mask)
 }
)
setMethod(
 f='torch_to_dense_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_to_dense_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_sparse_dim_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sparse_dim_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch__dimI_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch__dimI_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_dense_dim_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_dense_dim_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch__dimV_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch__dimV_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch__nnz_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch__nnz_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_coalesce_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_coalesce_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_is_coalesced_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_is_coalesced_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch__indices_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch__indices_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch__values_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch__values_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch__coalesced__',
 signature=list(self='externalptr', coalesced='logical'),
 definition=function(self, coalesced) {
torch__coalesced__2e9d8c347852e3f5e8b16b391c580e28(self, coalesced)
 }
)
setMethod(
 f='torch_indices_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_indices_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_values_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_values_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_numel_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_numel_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_unbind_',
 signature=list(self='externalptr', dim='numeric'),
 definition=function(self, dim) {
torch_unbind_ec971b768b5201c8c4ad6177a2be95ab(self, dim)
 }
)
setMethod(
 f='torch_to_sparse_',
 signature=list(self='externalptr', sparse_dim='numeric'),
 definition=function(self, sparse_dim) {
torch_to_sparse_b5eacf9bcb3f916a2b3b392bf8d0a876(self, sparse_dim)
 }
)
setMethod(
 f='torch_to_sparse_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_to_sparse_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_to_mkldnn_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_to_mkldnn_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_dequantize_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_dequantize_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_q_scale_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_q_scale_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_q_zero_point_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_q_zero_point_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_int_repr_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_int_repr_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_qscheme_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_qscheme_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_to_',
 signature=list(self='externalptr', options='list', non_blocking='logical', copy='logical'),
 definition=function(self, options, non_blocking, copy) {
torch_to_07fb7170362aa19d60b3bf80c613368b(self, options, non_blocking, copy)
 }
)
setMethod(
 f='torch_to_',
 signature=list(self='externalptr', device='Device', dtype='character', non_blocking='logical', copy='logical'),
 definition=function(self, device, dtype, non_blocking, copy) {
torch_to_90fe1befd83abd3f419da588b609ef89(self, device, dtype, non_blocking, copy)
 }
)
setMethod(
 f='torch_to_',
 signature=list(self='externalptr', dtype='character', non_blocking='logical', copy='logical'),
 definition=function(self, dtype, non_blocking, copy) {
torch_to_07b825f644ba168638538066d47424f6(self, dtype, non_blocking, copy)
 }
)
setMethod(
 f='torch_to_',
 signature=list(self='externalptr', other='externalptr', non_blocking='logical', copy='logical'),
 definition=function(self, other, non_blocking, copy) {
torch_to_4969670e973cd8d7b281281a8640d811(self, other, non_blocking, copy)
 }
)
setMethod(
 f='torch_item_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_item_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_set__',
 signature=list(self='externalptr', source='Storage'),
 definition=function(self, source) {
torch_set__3ed615ba1a472df076a4d9909a0cd30c(self, source)
 }
)
setMethod(
 f='torch_set__',
 signature=list(self='externalptr', source='Storage', storage_offset='numeric', size='numeric', stride='numeric'),
 definition=function(self, source, storage_offset, size, stride) {
torch_set__312dcae7f64e37bf3048b57f6468d8a1(self, source, storage_offset, size, stride)
 }
)
setMethod(
 f='torch_set__',
 signature=list(self='externalptr', source='externalptr'),
 definition=function(self, source) {
torch_set__aa628b77987871d31027b827e168238a(self, source)
 }
)
setMethod(
 f='torch_set__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_set__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_set_quantizer__',
 signature=list(self='externalptr', quantizer='ConstQuantizerPtr'),
 definition=function(self, quantizer) {
torch_set_quantizer__bade263e6781bf17b73374fb5be0b010(self, quantizer)
 }
)
setMethod(
 f='torch_is_set_to_',
 signature=list(self='externalptr', tensor='externalptr'),
 definition=function(self, tensor) {
torch_is_set_to_20e8a5d03011737e0a350fd8208fc32a(self, tensor)
 }
)
setMethod(
 f='torch_masked_fill__',
 signature=list(self='externalptr', mask='externalptr', value='numeric'),
 definition=function(self, mask, value) {
torch_masked_fill__3508ed7ed0d2c5cae6d9c0ce9b2fb2d4(self, mask, value)
 }
)
setMethod(
 f='torch_masked_fill_',
 signature=list(self='externalptr', mask='externalptr', value='numeric'),
 definition=function(self, mask, value) {
torch_masked_fill_41dc98a2ba7f8ad88de78079444f4bd8(self, mask, value)
 }
)
setMethod(
 f='torch_masked_fill__',
 signature=list(self='externalptr', mask='externalptr', value='externalptr'),
 definition=function(self, mask, value) {
torch_masked_fill__1af341297b28a33caffc2c17b883ba7b(self, mask, value)
 }
)
setMethod(
 f='torch_masked_fill_',
 signature=list(self='externalptr', mask='externalptr', value='externalptr'),
 definition=function(self, mask, value) {
torch_masked_fill_c4aae6cc2bd91e62fd5f083b1e8505a0(self, mask, value)
 }
)
setMethod(
 f='torch_masked_scatter__',
 signature=list(self='externalptr', mask='externalptr', source='externalptr'),
 definition=function(self, mask, source) {
torch_masked_scatter__8ecd133c55daf83edc61e4d6de3d48a0(self, mask, source)
 }
)
setMethod(
 f='torch_masked_scatter_',
 signature=list(self='externalptr', mask='externalptr', source='externalptr'),
 definition=function(self, mask, source) {
torch_masked_scatter_1769cd65a7836e60ad7642093f55a5d3(self, mask, source)
 }
)
setMethod(
 f='torch_view_',
 signature=list(self='externalptr', size='numeric'),
 definition=function(self, size) {
torch_view_6bf55218cd71b219c054293f38520051(self, size)
 }
)
setMethod(
 f='torch_put__',
 signature=list(self='externalptr', index='externalptr', source='externalptr', accumulate='logical'),
 definition=function(self, index, source, accumulate) {
torch_put__2b379e8493dddd64d5fcb57dfe46ab83(self, index, source, accumulate)
 }
)
setMethod(
 f='torch_index_add__',
 signature=list(self='externalptr', dim='numeric', index='externalptr', source='externalptr'),
 definition=function(self, dim, index, source) {
torch_index_add__0be4630fe46f5a3434059ed7debb5603(self, dim, index, source)
 }
)
setMethod(
 f='torch_index_add_',
 signature=list(self='externalptr', dim='numeric', index='externalptr', source='externalptr'),
 definition=function(self, dim, index, source) {
torch_index_add_073fd53276f8ca2975951ef418c77979(self, dim, index, source)
 }
)
setMethod(
 f='torch_index_fill__',
 signature=list(self='externalptr', dim='numeric', index='externalptr', value='numeric'),
 definition=function(self, dim, index, value) {
torch_index_fill__68bd4880045aa107467ff3395ce09125(self, dim, index, value)
 }
)
setMethod(
 f='torch_index_fill_',
 signature=list(self='externalptr', dim='numeric', index='externalptr', value='numeric'),
 definition=function(self, dim, index, value) {
torch_index_fill_6e7231b016b494e8f6fd347382bbf9dd(self, dim, index, value)
 }
)
setMethod(
 f='torch_index_fill__',
 signature=list(self='externalptr', dim='numeric', index='externalptr', value='externalptr'),
 definition=function(self, dim, index, value) {
torch_index_fill__928bbdb3d40ef89a9adaf7db52e85076(self, dim, index, value)
 }
)
setMethod(
 f='torch_index_fill_',
 signature=list(self='externalptr', dim='numeric', index='externalptr', value='externalptr'),
 definition=function(self, dim, index, value) {
torch_index_fill_d4b972e4d9c5472fc93caa76b85586c3(self, dim, index, value)
 }
)
setMethod(
 f='torch_scatter__',
 signature=list(self='externalptr', dim='numeric', index='externalptr', src='externalptr'),
 definition=function(self, dim, index, src) {
torch_scatter__a1889213e4ed60a091ae363b720154e4(self, dim, index, src)
 }
)
setMethod(
 f='torch_scatter_',
 signature=list(self='externalptr', dim='numeric', index='externalptr', src='externalptr'),
 definition=function(self, dim, index, src) {
torch_scatter_e1a8464ba19859b83be2b154413d58e4(self, dim, index, src)
 }
)
setMethod(
 f='torch_scatter__',
 signature=list(self='externalptr', dim='numeric', index='externalptr', value='numeric'),
 definition=function(self, dim, index, value) {
torch_scatter__68bd4880045aa107467ff3395ce09125(self, dim, index, value)
 }
)
setMethod(
 f='torch_scatter_',
 signature=list(self='externalptr', dim='numeric', index='externalptr', value='numeric'),
 definition=function(self, dim, index, value) {
torch_scatter_6e7231b016b494e8f6fd347382bbf9dd(self, dim, index, value)
 }
)
setMethod(
 f='torch_scatter_add__',
 signature=list(self='externalptr', dim='numeric', index='externalptr', src='externalptr'),
 definition=function(self, dim, index, src) {
torch_scatter_add__a1889213e4ed60a091ae363b720154e4(self, dim, index, src)
 }
)
setMethod(
 f='torch_scatter_add_',
 signature=list(self='externalptr', dim='numeric', index='externalptr', src='externalptr'),
 definition=function(self, dim, index, src) {
torch_scatter_add_e1a8464ba19859b83be2b154413d58e4(self, dim, index, src)
 }
)
setMethod(
 f='torch_lt__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_lt__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_lt__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_lt__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_gt__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_gt__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_gt__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_gt__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_le__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_le__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_le__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_le__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_ge__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_ge__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_ge__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_ge__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_eq__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_eq__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_eq__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_eq__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_ne__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_ne__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_ne__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_ne__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch___and___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___and___828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch___and___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___and___7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch___iand___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___iand___7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch___iand___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___iand___5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch___or___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___or___828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch___or___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___or___7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch___ior___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___ior___7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch___ior___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___ior___5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch___xor___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___xor___828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch___xor___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___xor___7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch___ixor___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___ixor___7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch___ixor___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___ixor___5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch___lshift___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___lshift___828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch___lshift___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___lshift___7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch___ilshift___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___ilshift___7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch___ilshift___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___ilshift___5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch___rshift___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___rshift___828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch___rshift___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___rshift___7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch___irshift___',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch___irshift___7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch___irshift___',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch___irshift___5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_lgamma__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_lgamma__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_atan2__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_atan2__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_tril__',
 signature=list(self='externalptr', diagonal='numeric'),
 definition=function(self, diagonal) {
torch_tril__5f49350cc06804354c82470a879c6411(self, diagonal)
 }
)
setMethod(
 f='torch_triu__',
 signature=list(self='externalptr', diagonal='numeric'),
 definition=function(self, diagonal) {
torch_triu__5f49350cc06804354c82470a879c6411(self, diagonal)
 }
)
setMethod(
 f='torch_digamma__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_digamma__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_polygamma__',
 signature=list(self='externalptr', n='numeric'),
 definition=function(self, n) {
torch_polygamma__5c9841a86a1bb2ecd6617444d7a089d2(self, n)
 }
)
setMethod(
 f='torch_erfinv__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_erfinv__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_renorm__',
 signature=list(self='externalptr', p='numeric', dim='numeric', maxnorm='numeric'),
 definition=function(self, p, dim, maxnorm) {
torch_renorm__9fae5d6a810b9bad032cf57f062d624f(self, p, dim, maxnorm)
 }
)
setMethod(
 f='torch_pow__',
 signature=list(self='externalptr', exponent='numeric'),
 definition=function(self, exponent) {
torch_pow__008abcd26c8445cd2a3cb5bcf2d286c7(self, exponent)
 }
)
setMethod(
 f='torch_pow__',
 signature=list(self='externalptr', exponent='externalptr'),
 definition=function(self, exponent) {
torch_pow__54cfc151844ae49e1de12914d9e69402(self, exponent)
 }
)
setMethod(
 f='torch_lerp__',
 signature=list(self='externalptr', end='externalptr', weight='numeric'),
 definition=function(self, end, weight) {
torch_lerp__24fc485413ad759980bbfddb3e995e6f(self, end, weight)
 }
)
setMethod(
 f='torch_lerp__',
 signature=list(self='externalptr', end='externalptr', weight='externalptr'),
 definition=function(self, end, weight) {
torch_lerp__f181da1a42d520a9a601e4f92273989b(self, end, weight)
 }
)
setMethod(
 f='torch_sign__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sign__77059e51c0f0414f12e11876ea1d8896(self)
 }
)
setMethod(
 f='torch_fmod__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_fmod__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_fmod__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_fmod__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_remainder__',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_remainder__7c9cc0828e809b4568149a32df76fd63(self, other)
 }
)
setMethod(
 f='torch_remainder__',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_remainder__5d476fcad960178d18feddbe371c5675(self, other)
 }
)
setMethod(
 f='torch_addbmm__',
 signature=list(self='externalptr', batch1='externalptr', batch2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, batch1, batch2, beta, alpha) {
torch_addbmm__e39679a201c809cf04d4768abd5f9472(self, batch1, batch2, beta, alpha)
 }
)
setMethod(
 f='torch_addbmm_',
 signature=list(self='externalptr', batch1='externalptr', batch2='externalptr', beta='numeric', alpha='numeric'),
 definition=function(self, batch1, batch2, beta, alpha) {
torch_addbmm_f084ab37fd7cee9fb89f911187c51117(self, batch1, batch2, beta, alpha)
 }
)
setMethod(
 f='torch_addcmul__',
 signature=list(self='externalptr', tensor1='externalptr', tensor2='externalptr', value='numeric'),
 definition=function(self, tensor1, tensor2, value) {
torch_addcmul__742865468d3d173af66a42cccde8e326(self, tensor1, tensor2, value)
 }
)
setMethod(
 f='torch_addcdiv__',
 signature=list(self='externalptr', tensor1='externalptr', tensor2='externalptr', value='numeric'),
 definition=function(self, tensor1, tensor2, value) {
torch_addcdiv__742865468d3d173af66a42cccde8e326(self, tensor1, tensor2, value)
 }
)
setMethod(
 f='torch_random__',
 signature=list(self='externalptr', from='numeric', to='numeric'),
 definition=function(self, from, to) {
torch_random__417777d0710e439fb989bef3f7645771(self, from, to)
 }
)
setMethod(
 f='torch_random__',
 signature=list(self='externalptr', to='numeric'),
 definition=function(self, to) {
torch_random__47742b4a683bcc7b28c10bc2cd1d8536(self, to)
 }
)
setMethod(
 f='torch_random__',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_random__4b76c0c01f02f75afc4f28e9c7848255(self)
 }
)
setMethod(
 f='torch_uniform__',
 signature=list(self='externalptr', from='double', to='double'),
 definition=function(self, from, to) {
torch_uniform__2f75be8f7c2310093195fe413f0445f8(self, from, to)
 }
)
setMethod(
 f='torch_normal__',
 signature=list(self='externalptr', mean='double', std='double'),
 definition=function(self, mean, std) {
torch_normal__56ea22ea9dddaf694df74da6a28fa826(self, mean, std)
 }
)
setMethod(
 f='torch_cauchy__',
 signature=list(self='externalptr', median='double', sigma='double'),
 definition=function(self, median, sigma) {
torch_cauchy__b607d43d206e68155367f854664713bf(self, median, sigma)
 }
)
setMethod(
 f='torch_log_normal__',
 signature=list(self='externalptr', mean='double', std='double'),
 definition=function(self, mean, std) {
torch_log_normal__56ea22ea9dddaf694df74da6a28fa826(self, mean, std)
 }
)
setMethod(
 f='torch_exponential__',
 signature=list(self='externalptr', lambd='double'),
 definition=function(self, lambd) {
torch_exponential__31e1c44de524960515a69c0dfbb817e2(self, lambd)
 }
)
setMethod(
 f='torch_geometric__',
 signature=list(self='externalptr', p='double'),
 definition=function(self, p) {
torch_geometric__c0d7479cf591e60afc5ecc0adace5091(self, p)
 }
)
setMethod(
 f='torch_diag_',
 signature=list(self='externalptr', diagonal='numeric'),
 definition=function(self, diagonal) {
torch_diag_3f1827a9d61846f9cc7b955f989a7c45(self, diagonal)
 }
)
setMethod(
 f='torch_cross_',
 signature=list(self='externalptr', other='externalptr', dim='numeric'),
 definition=function(self, other, dim) {
torch_cross_b8c427b31176d369b82935f8db13cb80(self, other, dim)
 }
)
setMethod(
 f='torch_triu_',
 signature=list(self='externalptr', diagonal='numeric'),
 definition=function(self, diagonal) {
torch_triu_3f1827a9d61846f9cc7b955f989a7c45(self, diagonal)
 }
)
setMethod(
 f='torch_tril_',
 signature=list(self='externalptr', diagonal='numeric'),
 definition=function(self, diagonal) {
torch_tril_3f1827a9d61846f9cc7b955f989a7c45(self, diagonal)
 }
)
setMethod(
 f='torch_trace_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_trace_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_ne_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_ne_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_ne_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_ne_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_eq_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_eq_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_eq_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_eq_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_ge_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_ge_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_ge_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_ge_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_le_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_le_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_le_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_le_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_gt_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_gt_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_gt_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_gt_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_lt_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_lt_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_lt_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_lt_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_take_',
 signature=list(self='externalptr', index='externalptr'),
 definition=function(self, index) {
torch_take_318ca949fbc9a40bb8a8115c70d08f69(self, index)
 }
)
setMethod(
 f='torch_index_select_',
 signature=list(self='externalptr', dim='numeric', index='externalptr'),
 definition=function(self, dim, index) {
torch_index_select_bb12de05582fff1d692ada6cd217b5d6(self, dim, index)
 }
)
setMethod(
 f='torch_masked_select_',
 signature=list(self='externalptr', mask='externalptr'),
 definition=function(self, mask) {
torch_masked_select_ed5bb55732ad3303430891d7fa131c57(self, mask)
 }
)
setMethod(
 f='torch_nonzero_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_nonzero_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_nonzero_numpy_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_nonzero_numpy_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_gather_',
 signature=list(self='externalptr', dim='numeric', index='externalptr', sparse_grad='logical'),
 definition=function(self, dim, index, sparse_grad) {
torch_gather_15abb7a6bd37d5a662e3afe129f5739f(self, dim, index, sparse_grad)
 }
)
setMethod(
 f='torch_addcmul_',
 signature=list(self='externalptr', tensor1='externalptr', tensor2='externalptr', value='numeric'),
 definition=function(self, tensor1, tensor2, value) {
torch_addcmul_60f230cf6615609d73b7013860ad4ace(self, tensor1, tensor2, value)
 }
)
setMethod(
 f='torch_addcdiv_',
 signature=list(self='externalptr', tensor1='externalptr', tensor2='externalptr', value='numeric'),
 definition=function(self, tensor1, tensor2, value) {
torch_addcdiv_60f230cf6615609d73b7013860ad4ace(self, tensor1, tensor2, value)
 }
)
setMethod(
 f='torch_lstsq_',
 signature=list(self='externalptr', A='externalptr'),
 definition=function(self, A) {
torch_lstsq_681b45f775987a9d4337e64ebe35838b(self, A)
 }
)
setMethod(
 f='torch_triangular_solve_',
 signature=list(self='externalptr', A='externalptr', upper='logical', transpose='logical', unitriangular='logical'),
 definition=function(self, A, upper, transpose, unitriangular) {
torch_triangular_solve_95047a6e651f1e1397e30787e0de642e(self, A, upper, transpose, unitriangular)
 }
)
setMethod(
 f='torch_symeig_',
 signature=list(self='externalptr', eigenvectors='logical', upper='logical'),
 definition=function(self, eigenvectors, upper) {
torch_symeig_917559e15993671449d66ba1008f876c(self, eigenvectors, upper)
 }
)
setMethod(
 f='torch_eig_',
 signature=list(self='externalptr', eigenvectors='logical'),
 definition=function(self, eigenvectors) {
torch_eig_1d467c833632d77e9e55713ef8c8b122(self, eigenvectors)
 }
)
setMethod(
 f='torch_svd_',
 signature=list(self='externalptr', some='logical', compute_uv='logical'),
 definition=function(self, some, compute_uv) {
torch_svd_2f217fcd9735b4f379dafcca6bdbb29e(self, some, compute_uv)
 }
)
setMethod(
 f='torch_cholesky_',
 signature=list(self='externalptr', upper='logical'),
 definition=function(self, upper) {
torch_cholesky_5af1d4d3ba3c95c341f2b7646d03f369(self, upper)
 }
)
setMethod(
 f='torch_cholesky_solve_',
 signature=list(self='externalptr', input2='externalptr', upper='logical'),
 definition=function(self, input2, upper) {
torch_cholesky_solve_025dfe9f0799032ef64ce3e4b4433d33(self, input2, upper)
 }
)
setMethod(
 f='torch_solve_',
 signature=list(self='externalptr', A='externalptr'),
 definition=function(self, A) {
torch_solve_681b45f775987a9d4337e64ebe35838b(self, A)
 }
)
setMethod(
 f='torch_cholesky_inverse_',
 signature=list(self='externalptr', upper='logical'),
 definition=function(self, upper) {
torch_cholesky_inverse_5af1d4d3ba3c95c341f2b7646d03f369(self, upper)
 }
)
setMethod(
 f='torch_qr_',
 signature=list(self='externalptr', some='logical'),
 definition=function(self, some) {
torch_qr_c656560b1e291e8f59e4c2f770446c1b(self, some)
 }
)
setMethod(
 f='torch_geqrf_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_geqrf_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_orgqr_',
 signature=list(self='externalptr', input2='externalptr'),
 definition=function(self, input2) {
torch_orgqr_ba4070bcaff2334005f02fd1ac15e6f7(self, input2)
 }
)
setMethod(
 f='torch_ormqr_',
 signature=list(self='externalptr', input2='externalptr', input3='externalptr', left='logical', transpose='logical'),
 definition=function(self, input2, input3, left, transpose) {
torch_ormqr_64c989218d6cdfc844b6d04abb9daab6(self, input2, input3, left, transpose)
 }
)
setMethod(
 f='torch_lu_solve_',
 signature=list(self='externalptr', LU_data='externalptr', LU_pivots='externalptr'),
 definition=function(self, LU_data, LU_pivots) {
torch_lu_solve_1b393e546fa15a4c4a3ff1ea45878c92(self, LU_data, LU_pivots)
 }
)
setMethod(
 f='torch_multinomial_',
 signature=list(self='externalptr', num_samples='numeric', replacement='logical'),
 definition=function(self, num_samples, replacement) {
torch_multinomial_58a43e013b869e223f72c31d9ed9864e(self, num_samples, replacement)
 }
)
setMethod(
 f='torch_lgamma_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_lgamma_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_digamma_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_digamma_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_polygamma_',
 signature=list(n='numeric', self='externalptr'),
 definition=function(n, self) {
torch_polygamma_a2138f85328db279ddde043403a417b3(n, self)
 }
)
setMethod(
 f='torch_erfinv_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_erfinv_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_dist_',
 signature=list(self='externalptr', other='externalptr', p='numeric'),
 definition=function(self, other, p) {
torch_dist_d0359fe95f05d18211bbcc94128382d4(self, other, p)
 }
)
setMethod(
 f='torch_atan2_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_atan2_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_lerp_',
 signature=list(self='externalptr', end='externalptr', weight='numeric'),
 definition=function(self, end, weight) {
torch_lerp_52119028f62f451b2f5688ea3610e128(self, end, weight)
 }
)
setMethod(
 f='torch_lerp_',
 signature=list(self='externalptr', end='externalptr', weight='externalptr'),
 definition=function(self, end, weight) {
torch_lerp_07603fb9e50ba263b5b23e7bb81db638(self, end, weight)
 }
)
setMethod(
 f='torch_histc_',
 signature=list(self='externalptr', bins='numeric', min='numeric', max='numeric'),
 definition=function(self, bins, min, max) {
torch_histc_bff667a73f52909f033d8c9de0081a1d(self, bins, min, max)
 }
)
setMethod(
 f='torch_sign_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_sign_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_fmod_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_fmod_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_fmod_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_fmod_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_remainder_',
 signature=list(self='externalptr', other='numeric'),
 definition=function(self, other) {
torch_remainder_828c724a305e4a254aee6141456d6fb1(self, other)
 }
)
setMethod(
 f='torch_remainder_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_remainder_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_min_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_min_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_min_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_min_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_max_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_max_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_max_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_max_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_median_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_median_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_sort_',
 signature=list(self='externalptr', dim='numeric', descending='logical'),
 definition=function(self, dim, descending) {
torch_sort_52a2acee48fe0d5321c4ec19c38c32d6(self, dim, descending)
 }
)
setMethod(
 f='torch_argsort_',
 signature=list(self='externalptr', dim='numeric', descending='logical'),
 definition=function(self, dim, descending) {
torch_argsort_52a2acee48fe0d5321c4ec19c38c32d6(self, dim, descending)
 }
)
setMethod(
 f='torch_topk_',
 signature=list(self='externalptr', k='numeric', dim='numeric', largest='logical', sorted='logical'),
 definition=function(self, k, dim, largest, sorted) {
torch_topk_92a7e3ac7fad4d88b3448c70a70c3f8a(self, k, dim, largest, sorted)
 }
)
setMethod(
 f='torch_all_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_all_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_any_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_any_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
setMethod(
 f='torch_renorm_',
 signature=list(self='externalptr', p='numeric', dim='numeric', maxnorm='numeric'),
 definition=function(self, p, dim, maxnorm) {
torch_renorm_56a4ac7915d3aee407c8286c79a259ff(self, p, dim, maxnorm)
 }
)
setMethod(
 f='torch_unfold_',
 signature=list(self='externalptr', dimension='numeric', size='numeric', step='numeric'),
 definition=function(self, dimension, size, step) {
torch_unfold_020977f59530963a27cd1039400e50c9(self, dimension, size, step)
 }
)
setMethod(
 f='torch_equal_',
 signature=list(self='externalptr', other='externalptr'),
 definition=function(self, other) {
torch_equal_7f12b0bdc5e3c8186277ce82f15149b2(self, other)
 }
)
setMethod(
 f='torch_pow_',
 signature=list(self='externalptr', exponent='externalptr'),
 definition=function(self, exponent) {
torch_pow_b69e041157596e34bdef3154d84f2d1e(self, exponent)
 }
)
setMethod(
 f='torch_alias_',
 signature=list(self='externalptr'),
 definition=function(self) {
torch_alias_68396f1df3a98eb80570d6202c3c8b18(self)
 }
)
