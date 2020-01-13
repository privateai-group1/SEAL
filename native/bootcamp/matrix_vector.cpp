#include "matrix_vector.h"

using namespace std;
using namespace seal;

matrix random_square_matrix(size_t dim) {
	matrix M(dim);
	for (size_t i = 0; i < M.size(); i++) {
		M[i].resize(dim);
		for (size_t j = 0; j < dim; j++) {
			M[i][j] = (double)rand() / RAND_MAX;
		}
	}
	return M;
}

matrix identity_matrix(size_t dim) {
	matrix M(dim);
	for (size_t i = 0; i < M.size(); i++) {
		M[i].resize(dim);
		M[i][i] = 1;
	}
	return M;
}

vec random_vector(size_t dim) {
	vec v(dim);
	for (size_t j = 0; j < dim; j++) {
		v[j] = (double)rand() / RAND_MAX;
	}
	return v;
}

vec mvp(matrix M, vec v) {
	vec Mv(M.size(), 0);
	for (size_t i = 0; i < M.size(); i++) {
		if (v.size() != M[0].size()) {
			throw new invalid_argument("Vector and Matrix dimension not compatible.");
		}
		else {
			for (size_t j = 0; j < M[0].size(); j++) {
				Mv[i] += M[i][j] * v[j];
			}
		}
	}
	return Mv;
}

void ptxt_matrix_enc_vector_product(const GaloisKeys& galois_keys, Evaluator& evaluator,
	vector<Plaintext> ptxt_diags, const Ciphertext& ctv, Ciphertext& enc_result, size_t dim) {
	// TODO: Make this aware of batching, i.e. do not include non-relevant slots into computation
	// TODO: Switch this to the bs-GS algorithm!
	Ciphertext temp;
	if (dim == 256) {
		// baby-stp giant-step
		int n1 = 16;
		int n2 = 16;
		for (int k = 0; k < n2; ++k) {
			Ciphertext inner;
			for (int j = 0; j < n1; ++j) {
				evaluator.rotate_vector(ctv, j, galois_keys, temp);
				evaluator.mod_switch_to_inplace(ptxt_diags[k * n1 + j], temp.parms_id());
				evaluator.multiply_plain_inplace(temp, ptxt_diags[k * n1 + j]);
				if (j == 0) {
					inner = temp;
				}
				else {
					//TODO: This should probably use 3-for-2 addition
					evaluator.add_inplace(inner, temp);
				}

			}
			evaluator.rotate_vector_inplace(inner, k * n1, galois_keys);
			if (k == 0) {
				enc_result = inner;
			}
			else {
				evaluator.add_inplace(enc_result, inner);
			}
		}


	}
	else {

		for (int i = 0; i < dim; i++) {
			// rotate 
			evaluator.rotate_vector(ctv, i, galois_keys, temp);
			// multiply
			evaluator.mod_switch_to_inplace(ptxt_diags[i], temp.parms_id());
			evaluator.multiply_plain_inplace(temp, ptxt_diags[i]);
			if (i == 0) {
				enc_result = temp;
			}
			else { //TODO: This should probably use 3-for-2 addition
				evaluator.add_inplace(enc_result, temp);
			}
		}
	}

}

/// Diagonal of a !!square!! matrix M
vec diag(matrix M, size_t d) {
	//TODO: If not square, or not large enough to have d-th diagonal, then throw invalid_argument
	size_t dim = M.size();
	vec diag(dim);
	for (size_t i = 0; i < dim; i++) {
		diag[i] = M[i][(i + d) % dim];
	}
	return diag;
}

vector<vec> diags(matrix M) {
	vector<vec> diags(M.size());
	for (size_t i = 0; i < M.size(); ++i) {
		diags[i] = diag(M, i);
	}
	return diags;
}

matrix add(matrix A, matrix B) {
	if (A.size() != B.size() || (A.size() > 0 && A[0].size() != B[0].size())) {
		throw new invalid_argument("Matrices must have the same dimensions.");
	}
	else {
		matrix C(A.size());
		for (size_t i = 0; i < A.size(); i++) {
			C[i].resize(A[0].size());
			for (size_t j = 0; j < A[0].size(); j++) {
				C[i][j] = A[i][j] + B[i][j];
			}
		}
		return C;
	}
}

vec add(vec a, vec b) {
	if (a.size() != b.size()) {
		throw new invalid_argument("Vectors must have the same dimensions.");
	}
	else {
		vec c(a.size());
		for (size_t i = 0; i < a.size(); i++) {
			c[i] = a[i] + b[i];
		}
		return c;
	}
}