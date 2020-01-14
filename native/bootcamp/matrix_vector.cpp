#include "matrix_vector.h"

using namespace std;
using namespace seal;

matrix random_square_matrix(size_t dim)
{
	matrix M(dim);
	for (size_t i = 0; i < M.size(); i++)
	{
		M[i].resize(dim);
		for (size_t j = 0; j < dim; j++)
		{
			M[i][j] = static_cast<double>(rand()) / RAND_MAX;
		}
	}
	return M;
}

matrix identity_matrix(size_t dim)
{
	matrix M(dim);
	for (size_t i = 0; i < M.size(); i++)
	{
		M[i].resize(dim);
		M[i][i] = 1;
	}
	return M;
}

vec random_vector(size_t dim)
{
	vec v(dim);
	for (size_t j = 0; j < dim; j++)
	{
		v[j] = static_cast<double>(rand()) / RAND_MAX;
	}
	return v;
}

vec mvp(matrix M, vec v)
{
	if (M.size() == 0)
	{
		throw invalid_argument("Matrix must be well formed and non-zero-dimensional");
	}

	vec Mv(M.size(), 0);
	for (size_t i = 0; i < M.size(); i++)
	{
		if (v.size() != M[0].size())
		{
			throw invalid_argument("Vector and Matrix dimension not compatible.");
		}
		else
		{
			for (size_t j = 0; j < M[0].size(); j++)
			{
				Mv[i] += M[i][j] * v[j];
			}
		}
	}
	return Mv;
}


matrix add(matrix A, matrix B)
{
	if (A.size() != B.size() || (A.size() > 0 && A[0].size() != B[0].size()))
	{
		throw invalid_argument("Matrices must have the same dimensions.");
	}
	else
	{
		matrix C(A.size());
		for (size_t i = 0; i < A.size(); i++)
		{
			C[i].resize(A[0].size());
			for (size_t j = 0; j < A[0].size(); j++)
			{
				C[i][j] = A[i][j] + B[i][j];
			}
		}
		return C;
	}
}

vec add(vec a, vec b)
{
	if (a.size() != b.size())
	{
		throw invalid_argument("Vectors must have the same dimensions.");
	}
	else
	{
		vec c(a.size());
		for (size_t i = 0; i < a.size(); i++)
		{
			c[i] = a[i] + b[i];
		}
		return c;
	}
}

vec diag(matrix M, size_t d)
{
	const size_t dim = M.size();
	if (dim == 0 || M[0].size() != dim || d >= dim)
	{
		throw invalid_argument("Matrix must be square and d must be smaller than matrix dimension.");
	}
	vec diag(dim);
	for (size_t i = 0; i < dim; i++)
	{
		diag[i] = M[i][(i + d) % dim];
	}
	return diag;
}

vector<vec> diagonals(const matrix M)
{
	if (M.size() == 0)
	{
		throw invalid_argument("Matrix must be square and have non-zero dimension.");
	}
	vector<vec> diagonals(M.size());
	for (size_t i = 0; i < M.size(); ++i)
	{
		diagonals[i] = diag(M, i);
	}
	return diagonals;
}

vec duplicate(const vec v)
{
	size_t dim = v.size();
	vec r;
	r.reserve(2 * dim);
	r.insert(r.begin(), v.begin(), v.end());
	r.insert(r.end(), v.begin(), v.end());
	return r;
}

vec mvp_from_diagonals(std::vector<vec> diagonals, vec v)
{
	const size_t dim = diagonals.size();
	if (dim == 0 || diagonals[0].size() != dim || v.size() != dim)
	{
		throw invalid_argument("Matrix must be square, Matrix and vector must have matching non-zero dimension.");
	}
	vec r(dim);
	for (size_t i = 0; i < dim; ++i)
	{
		// t = diagonals[i] * v, component wise
		vec t(dim);
		for (size_t j = 0; j < dim; ++j)
		{
			t[j] = diagonals[i][j] * v[j];
		}

		// Accumulate result
		r = add(r, t);

		// Rotate v to next position (at the end, because it needs to be un-rotated for first iteration)
		rotate(v.begin(), v.begin() + 1, v.end());
	}
	return r;
}


void ptxt_matrix_enc_vector_product(const GaloisKeys& galois_keys, Evaluator& evaluator,
                                    size_t dim, vector<Plaintext> ptxt_diagonals, const Ciphertext& ctv,
                                    Ciphertext& enc_result)
{
	// TODO: Make this aware of batching, i.e. do not include non-relevant slots into computation
	// TODO: Implement generic baby-step giant-step for dim other than 256
	Ciphertext temp;
	if (dim == 256)
	{
		// baby-step giant-step
		size_t n1 = 16;
		size_t n2 = 16;
		for (size_t k = 0; k < n2; ++k)
		{
			Ciphertext inner;
			for (int j = 0; j < n1; ++j)
			{
				evaluator.rotate_vector(ctv, j, galois_keys, temp);
				evaluator.mod_switch_to_inplace(ptxt_diagonals[k * n1 + j], temp.parms_id());
				evaluator.multiply_plain_inplace(temp, ptxt_diagonals[k * n1 + j]);
				if (j == 0)
				{
					inner = temp;
				}
				else
				{
					//TODO: This should probably use 3-for-2 addition
					evaluator.add_inplace(inner, temp);
				}
			}
			evaluator.rotate_vector_inplace(inner, k * n1, galois_keys);
			if (k == 0)
			{
				enc_result = inner;
			}
			else
			{
				evaluator.add_inplace(enc_result, inner);
			}
		}
	}
	else
	{
		for (int i = 0; i < dim; i++)
		{
			// rotate 
			evaluator.rotate_vector(ctv, i, galois_keys, temp);
			// multiply
			evaluator.mod_switch_to_inplace(ptxt_diagonals[i], temp.parms_id());
			evaluator.multiply_plain_inplace(temp, ptxt_diagonals[i]);
			if (i == 0)
			{
				enc_result = temp;
			}
			else
			{
				//TODO: This should probably use 3-for-2 addition
				evaluator.add_inplace(enc_result, temp);
			}
		}
	}
}
