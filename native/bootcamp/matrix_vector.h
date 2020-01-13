#pragma once
#include <vector>
#include "seal/seal.h"

typedef std::vector<std::vector<double>> matrix;
typedef std::vector<double> vec;

matrix random_square_matrix(size_t dim);

matrix identity_matrix(size_t dim);

vec random_vector(size_t dim);

vec mvp(matrix M, vec v);

void ptxt_matrix_enc_vector_product(const seal::GaloisKeys& galois_keys, seal::Evaluator& evaluator,
	std::vector<seal::Plaintext> ptxt_diags, const seal::Ciphertext& ctv, seal::Ciphertext& enc_result, size_t dim);

/// Diagonal of a !!square!! matrix M
vec diag(matrix M, size_t d);

std::vector<vec> diags(matrix M);

matrix add(matrix A, matrix B);

vec add(vec a, vec b);