#include "example_rnn.h"

using namespace std;
using namespace seal;


typedef vector<vector<double>> matrix;
typedef vector<double> vec;

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
	for (int i = 0; i < dim; i++) {
		// rotate 
		evaluator.rotate_vector(ctv, i, galois_keys, temp);
		// multiply
		evaluator.multiply_plain_inplace(temp, ptxt_diags[i]);
		if (i == 0) {
			enc_result = temp;
		}
		else { //TODO: This should probably use 3-for-2 addition
			evaluator.add_inplace(enc_result, temp);
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


void example_rnn()
{
	// Setup Parameters
	EncryptionParameters params(scheme_type::CKKS);

	vector<int> moduli = { 50, 40, 40, 40, 40, 59 }; //TODO: Select proper moduli
	size_t poly_modulus_degree = 16384; // TODO: Select appropriate degree
	double scale = pow(2.0, 40); //TODO: Select appropriate scale

	params.set_poly_modulus_degree(poly_modulus_degree);
	params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, moduli));
	auto context = SEALContext::Create(params);
	print_parameters(context);

	// Client-side
	cout << "------------- CLIENT ------------------" << endl;

	// Generate Keys
	cout << "Generating keys...";
	KeyGenerator keygen(context);
	auto public_key = keygen.public_key();
	auto secret_key = keygen.secret_key();
	auto relin_keys = keygen.relin_keys();
	{
		ofstream fs("rnn.galk", ios::binary);
		keygen.galois_keys_save(fs); //TODO: Generate only required galois keys
	}

	Encryptor encryptor(context, public_key);
	encryptor.set_secret_key(secret_key);
	Decryptor decryptor(context, secret_key);
	CKKSEncoder encoder(context);
	cout << "...done " << endl;

	/// dimension of hidden thingy, also dimension of word embeddings for square-ness of matrices
	size_t ml_dim = 100;
	/// Number of words in sentence
	size_t num_words = 10;

	// Secret input - represented as doubled thingy
	vector<vec> xxs(num_words);
	for (size_t i = 0; i < xxs.size(); ++i) {
		xxs[i] = random_vector(ml_dim);
	}
	print_matrix(xxs, "inputs x_i, by row");

	vec xs(2 * num_words * ml_dim);
	for (size_t i = 0; i < xs.size(); ++i) {
		xs[i] = xxs[i / (2 * ml_dim)][i % ml_dim];
	}

	cout << "Encoding and encrypting input...";
	Plaintext xs_ptxt;
	encoder.encode(xs, scale, xs_ptxt);
	{
		ofstream fs("xs.ct", ios::binary);
		encryptor.encrypt_symmetric_save(xs_ptxt, fs);
	}
	cout << "...done" << endl;

	// SERVER SIDE:
	cout << "------------- SERVER ------------------" << endl;

	// Load Galois Keys
	cout << "Loading Galois keys...";
	GaloisKeys galk;
	{
		ifstream fs("rnn.galk", ios::binary);
		galk.load(context, fs);
	}
	cout << "...done" << endl;

	// Load Ciphertext
	cout << "Loading ciphertext...";
	Ciphertext xs_ctxt;
	{
		ifstream fs("xs.ct", ios::binary);
		xs_ctxt.load(context, fs);
	}
	cout << "...done" << endl;

	// Create the Evaluator
	Evaluator evaluator(context);

	// Weight matrices for encoding phase
	auto W_x = identity_matrix(ml_dim); // TODO: READD random_square_matrix(ml_dim);
	print_matrix(W_x, "W_x:");
	auto W_h = identity_matrix(ml_dim); // TODO: READD random_square_matrix(ml_dim);
	print_matrix(W_h, "W_h:");
	// Represent weight matrices diagonally
	vector<vec> diags_W_x = diags(W_x);
	print_matrix(diags_W_x, "diags_W_x:");
	vector<vec> diags_W_h = diags(W_h);
	print_matrix(diags_W_h, "diags_W_h:");

	// Weight vectors/matrices for decoding phase
	auto b = random_vector(ml_dim);
	auto W = random_square_matrix(ml_dim);
	auto U = random_square_matrix(ml_dim);
	auto V = random_square_matrix(ml_dim);
	auto c = random_vector(ml_dim);
	// Represent weight matrices diagonally
	vector<vec> diags_W = diags(W);
	vector<vec> diags_U = diags(U);
	vector<vec> diags_V = diags(V);

	// Start thingies?
	auto s_x = random_vector(ml_dim);
	print_vector(s_x, "s_x:");
	auto s_h = random_vector(ml_dim);
	print_vector(s_h, "s_h:");


	// Encode diags as ptxts
	cout << "Encoding plaintext model...";
	// TODO: Do this at appropriate scale when required
	vector<Plaintext>
		ptxt_diags_W_x(ml_dim),
		ptxt_diags_W_h(ml_dim),
		ptxt_diags_W(ml_dim),
		ptxt_diags_U(ml_dim),
		ptxt_diags_V(ml_dim);
	for (size_t i = 0; i < ml_dim; ++i) {
		encoder.encode(diags_W_x[i], scale, ptxt_diags_W_x[i]);
		encoder.encode(diags_W_h[i], scale, ptxt_diags_W_h[i]);
		encoder.encode(diags_W[i], scale, ptxt_diags_W[i]);
		encoder.encode(diags_U[i], scale, ptxt_diags_U[i]);
		encoder.encode(diags_V[i], scale, ptxt_diags_V[i]);
	}
	cout << "...done" << endl;

	// Encoding Phase
	cout << "Starting encoding phase of RNN:" << endl;

	// Compute W_x * x_1 for the first block
	cout << "Computing W_x * x_1...";
	Ciphertext h1_ctxt;
	ptxt_matrix_enc_vector_product(galk, evaluator, ptxt_diags_W_x, xs_ctxt, h1_ctxt, ml_dim);
	cout << "...done" << endl;

	// Compute encrypted result:
	{
		Plaintext ptxt_block1;
		decryptor.decrypt(h1_ctxt, ptxt_block1);
		vec block1;
		encoder.decode(ptxt_block1, block1);
		cout << "Encrypted result:" << endl;
		print_vector(vector(block1.begin(), block1.begin() + ml_dim));
	}

	// Compute expected result:
	{
		vec x1 = vec(xs.begin(), xs.begin() + ml_dim);
		vec r = mvp(W_x, x1);
		cout << "Expected result: " << endl;
		print_vector(r);
	}

	// Compute W_h * h_0 and add to previous computed W_x * x_1
	cout << "Compute (ptxt) W_h * h_0 and add to previously computed (W_x * x_1)...";
	auto h_0 = vec(ml_dim); //TODO: READD add(mvp(W_x, s_x), mvp(W_h, s_h));
	auto rhs_1 = mvp(W_h, h_0);
	Plaintext ptxt_rhs1;
	encoder.encode(rhs_1, h1_ctxt.scale(), ptxt_rhs1);
	evaluator.add_plain_inplace(h1_ctxt, ptxt_rhs1);
	cout << "...done" << endl;

	// Compute expected result:
	{
		print_vector(h_0, "h_0 = W_x * s_x:");
		print_vector(rhs_1, "rhs_1 = W_h * h_0:");
		vec x1 = vec(xs.begin(), xs.begin() + ml_dim);
		vec r = add(rhs_1, mvp(W_x, x1));
		cout << "Expected result: " << endl;
		print_vector(vector(r.begin(), r.begin() + ml_dim));
	}

	// Compute encrypted result:
	{
		Plaintext ptxt_h1;
		decryptor.decrypt(h1_ctxt, ptxt_h1);
		vec block1;
		encoder.decode(ptxt_h1, block1);
		cout << "Encrypted result:" << endl;
		print_vector(block1);
	}

	// TODO: Compute W_x * x_i + W_h * h_i-1 for the remaining blocks
	Ciphertext h = h1_ctxt;
	Ciphertext tmp_whh;
	Ciphertext tmp_wxx;
	Ciphertext xs_rot;
	for (size_t i = 2; i <= num_words; ++i) {
		// Compute W_h * h_(i-1)
		cout << "Compute W_h * h_" << i - 1 << "...";
		ptxt_matrix_enc_vector_product(galk, evaluator, ptxt_diags_W_h, h, tmp_whh, ml_dim);
		cout << "...done" << endl;

		// Compute W_x * x_i
		cout << "Compute W_x * x_" << i;
		evaluator.rotate_vector(xs_ctxt, 2 * ml_dim, galk, xs_rot); //TODO: w_x * x_i from batching
		ptxt_matrix_enc_vector_product(galk, evaluator, ptxt_diags_W_x, xs_rot, tmp_wxx, ml_dim);
		cout << "...done" << endl;

		// h_i = (W_x * x_i) + (W_h * h_(i-1))
		cout << "Add together to form h_" << i;

		//TODO: Is this rescaling safe?
		evaluator.rescale_to_next_inplace(tmp_whh);
		evaluator.mod_switch_to_inplace(tmp_wxx, tmp_whh.parms_id());
		tmp_whh.scale() = tmp_wxx.scale();
		
		evaluator.add(tmp_wxx, tmp_whh, h);
		cout << "...done" << endl;
	}


	// TODO: Decoding Phase


}