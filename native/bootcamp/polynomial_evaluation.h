#pragma once

#include "helpers.h"
#include "seal/seal.h"

void example_polyeval();

void dot_product(std::vector<seal::Plaintext>& pts, int skip, const std::vector<seal::Ciphertext>& ctx, seal::Evaluator& evaluator, seal::Ciphertext& destination);

void compute_all_powers(const seal::Ciphertext& ctx, int degree, seal::Evaluator& evaluator, seal::RelinKeys& relin_keys, std::vector<seal::Ciphertext>& powers);
