#include "gtest/gtest.h"
#include "matrix_vector_crypto.h"

using namespace std;
using namespace seal;

namespace MVCryptoTests {

	/**
	 * \brief Helper function to test plaintext-matrix-encrypted-vector products.
	 * \param dimension Length of vector and dimension of matrix
	 * \param bsgs Whether or not to use the baby-step giant-step algorithm
	 */
	void MatrixVectorProductTest(size_t dimension, bool bsgs = false)
	{
		const auto m = random_square_matrix(dimension);
		const auto v = random_vector(dimension);
		const auto expected = mvp(m, v);

		// Setup SEAL Parameters
		EncryptionParameters params(scheme_type::CKKS);
		const double scale = pow(2.0, 40);
		params.set_poly_modulus_degree(8192);
		params.set_coeff_modulus(CoeffModulus::Create(8192, { 50, 40, 50 }));
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
		vector<Plaintext> ptxt_diagonals(dimension);
		for (size_t i = 0; i < dimension; ++i)
		{
			encoder.encode(diag(m, i), scale, ptxt_diagonals[i]);
		}

		// Decode and compare
		for (size_t i = 0; i < dimension; ++i)
		{
			vec t;
			encoder.decode(ptxt_diagonals[i], t);
			t.resize(dimension);
			for (size_t j = 0; j < dimension; ++j)
			{
				// Test if value is within 0.1% of the actual value or 10 sig figs
				EXPECT_NEAR(t[j], diag(m, i)[j], max(0.000000001, 0.001 * diag(m, i)[j]));
			}
		}

		// Encrypt vector
		Plaintext ptxt_v;

		// Do we need to duplicate elements in the diagonals vectors during encoding to ensure meaningful rotations?
		if ((params.poly_modulus_degree() / 2) != dimension) {
			encoder.encode(duplicate(v), pow(2.0, 40), ptxt_v);
		}
		else
		{
			encoder.encode(v, pow(2.0, 40), ptxt_v);
		}
		Ciphertext ctxt_v;
		encryptor.encrypt_symmetric(ptxt_v, ctxt_v);

		// Decrypt and compare
		// TODO: Decrypt and check vector comes out alright.

		// Compute MVP
		Ciphertext ctxt_r;
		if (bsgs)
		{
			ptxt_matrix_enc_vector_product_bsgs(galois_keys, evaluator, encoder, dimension, diagonals(m), ctxt_v, ctxt_r);
		}
		else
		{
			ptxt_matrix_enc_vector_product(galois_keys, evaluator, dimension, ptxt_diagonals, ctxt_v, ctxt_r);
		}


		// Decrypt and decode result
		Plaintext ptxt_r;
		decryptor.decrypt(ctxt_r, ptxt_r);
		vec r;
		encoder.decode(ptxt_r, r);
		r.resize(dimension);

		for (size_t i = 0; i < dimension; ++i)
		{
			// Test if value is within 0.1% of the actual value or 10 sig figs
			EXPECT_NEAR(r[i], expected[i], max(0.000000001, 0.001 * expected[i]));
		}

		//TODO: The EXPECT_FLOAT_EQ assertions might occasionally fail since the noise is somewhat random and we get less than 32 bits of guaranteed precision from these parameters
	}

	TEST(EncryptedMVP, MatrixVectorProduct_15)
	{
		MatrixVectorProductTest(15);
	}

	TEST(EncryptedMVP, MatrixVectorProduct_256)
	{
		MatrixVectorProductTest(256);
	}

	TEST(EncryptedMVP, MatrixVectorProductBSGS_15)
	{
		// BSGS currently only supports square-number  dimensions
		EXPECT_THROW(MatrixVectorProductTest(15, true), invalid_argument);
	}

	TEST(EncryptedMVP, MatrixVectorProductBSGS_4)
	{
		MatrixVectorProductTest(4, true);
	}

	TEST(EncryptedMVP, MatrixVectorProductBSGS_16)
	{
		MatrixVectorProductTest(16, true);
	}

	TEST(EncryptedMVP, MatrixVectorProductBSGS_49)
	{
		MatrixVectorProductTest(49, true);
	}
	TEST(EncryptedMVP, MatrixVectorProductBSGS_256)
	{
		MatrixVectorProductTest(256, true);
	}

	// This test would be nice to have, but takes an unreasonably long time to complete on a desktop PC
	// TEST(EncryptedMVP, MatrixVectorProductBSGS_exact_slots)
	// {
	// 	MatrixVectorProductTest(4096, true);
	// }

	TEST(EncryptedMVP, MatrixVectorProductBSGS_5000)
	{
		// Since this is neither the number of slots, nor does it fit if duplicated, this should fail
		EXPECT_THROW(MatrixVectorProductTest(5000, true), invalid_argument);
	}
}