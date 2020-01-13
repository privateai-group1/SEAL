#include "gtest/gtest.h"
#include "matrix_vector.h"

using namespace std;
using namespace seal;

const size_t dim = 15;
TEST(Generation, RandomMatrix)
{
	const auto t = random_square_matrix(15);
	EXPECT_EQ(t.size(), dim);
	for (auto& x : t)
	{
		EXPECT_EQ(x.size(), dim);

		for (auto& y : x)
		{
			EXPECT_TRUE((0 <= y) && (y <= 1));
		}
	}
}

TEST(Generation, IdentityMatrix)
{
	const auto t = identity_matrix(15);
	EXPECT_EQ(t.size(), dim);
	for (size_t i = 0; i < t.size(); ++i)
	{
		EXPECT_EQ(t[i].size(), dim);
		for (size_t j = 0; j < t[i].size(); ++j)
		{
			EXPECT_EQ(t[i][j], (i == j));
		}
	}
}

TEST(Generation, RandomVector)
{
	const auto t = random_vector(15);
	EXPECT_EQ(t.size(), dim);
	for (auto& x : t)
	{
		EXPECT_TRUE((0 <= x) && (x <= 1));
	}
}

TEST(PlaintextOperations, MatrixVectorProduct)
{
	const auto m = random_square_matrix(dim);
	const auto v = random_vector(dim);

	// Standard multiplication
	const auto r = mvp(m, v);	
	EXPECT_EQ(r.size(), dim);
	for (size_t i = 0; i < dim; ++i)
	{
		double sum = 0;
		for(size_t j = 0; j < dim; ++j)
		{
			sum += v[j] * m[i][j];
		}
		EXPECT_EQ(r[i], sum);
	}

	// identity matrix should be doing identity things
	const auto id = identity_matrix(dim);
	EXPECT_EQ(v, mvp(id, v));

	// Mismatching sizes should throw exception
	EXPECT_THROW(mvp(m, {}), invalid_argument);
	EXPECT_THROW(mvp({}, v), invalid_argument);
}

TEST(PlaintextOperations, MatrixAdd)
{
	const auto m1 = random_square_matrix(dim);
	const auto m2 = random_square_matrix(dim);

	// Standard addition
	const auto r = add(m1, m2);
	EXPECT_EQ(r.size(), dim);
	for (size_t i = 0; i < dim; ++i)
	{
		EXPECT_EQ(r[i].size(), dim);
		for (size_t j = 0; j < dim; ++j)
		{
			EXPECT_EQ(r[i][j], (m1[i][j] + m2[i][j]));
		}
	}

	// Mismatched sizes
	EXPECT_THROW(add({}, m2), invalid_argument);
	EXPECT_THROW(add(m1, {}),invalid_argument);
	EXPECT_THROW(add(matrix(dim), m2), invalid_argument);
	EXPECT_THROW(add(m1,matrix(dim)), invalid_argument);
}

TEST(PlaintextOperations, VectorAdd)
{
	const auto v1 = random_vector(dim);
	const auto v2 = random_vector(dim);

	// Standard addition
	const auto r = add(v1, v2);
	EXPECT_EQ(r.size(), dim);
	for (size_t i = 0; i < dim; ++i)
	{
		EXPECT_EQ(r[i], (v1[i] + v2[i]));
	}

	// Mismatched sizes
	EXPECT_THROW(add({}, v2), invalid_argument);
	EXPECT_THROW(add(v1, {}), invalid_argument);
}

TEST(PlaintextOperations, Diag)
{
	const auto m = random_square_matrix(dim);
	for (size_t d = 0; d < dim; ++d)
	{
		const auto r = diag(m, d);
		EXPECT_EQ(r.size(), dim);
		for(size_t i = 0; i < dim; ++i)
		{
			EXPECT_EQ(r[i], m[i][(i+d) % dim]);
		}
	}

	// Non-square
	EXPECT_THROW(diag(matrix(dim),0), invalid_argument);

	// Non-existent diagonal
	EXPECT_THROW(diag(m, dim + 1), invalid_argument);
}

TEST(PlaintextOperations, Diagonals)
{
	const auto m = random_square_matrix(dim);
	const auto r = diagonals(m);
	EXPECT_EQ(r.size(), dim);
	for (size_t d = 0; d < dim; ++d)
	{		
		EXPECT_EQ(r[d].size(), dim);
		for (size_t i = 0; i < dim; ++i)
		{
			EXPECT_EQ(r[d][i], m[i][(i + d) % dim]);
		}
	}

	// Non-square
	EXPECT_THROW(diag(matrix(dim), 0), invalid_argument);
}

TEST(EncryptedMVP,MatrixVectorProduct)
{
	const auto m = random_square_matrix(dim);
	const auto v = random_vector(dim);
	const auto expected = mvp(m, v);

	// Setup SEAL Parameters
	EncryptionParameters params(scheme_type::CKKS);
	params.set_poly_modulus_degree(8192);
	params.set_coeff_modulus(CoeffModulus::Create(8192, {40,40,40,50}));
	auto context = SEALContext::Create(params);
	

	// Generate required keys
	KeyGenerator keygen(context);
	auto public_key = keygen.public_key();
	auto secret_key = keygen.secret_key();
	auto relin_keys = keygen.relin_keys();
	auto galois_keys = keygen.galois_keys();

	Encryptor encryptor(context, public_key);
	encryptor.set_secret_key(secret_key);
	Decryptor decryptor(context, secret_key);
	CKKSEncoder encoder(context);
	Evaluator evaluator(context);

	// Encode matrix
	vector<Plaintext > ptxt_diagonals(dim);
	for (size_t i = 0; i < dim; ++i) {
		encoder.encode(diag(m,i), pow(2.0, 40), ptxt_diagonals[i]);
	}

	// Encrypt vector
	Plaintext ptxt_v;
	encoder.encode(v, pow(2.0, 40), ptxt_v);
	Ciphertext ctxt_v;
	encryptor.encrypt_symmetric(ptxt_v, ctxt_v);

	// Compute MVP
	Ciphertext ctxt_r;
	ptxt_matrix_enc_vector_product(galois_keys, evaluator, dim, ptxt_diagonals, ctxt_v, ctxt_r);

	// Decrypt and decode result
	Plaintext ptxt_r;
	decryptor.decrypt(ctxt_r, ptxt_r);
	vec r;
	encoder.decode(ptxt_r, r);
	r.resize(dim);

	EXPECT_EQ(r, expected);
	
}