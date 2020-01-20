#include "example_rnn.h"
#include "helpers.h"
#include "timer.h"
#include "matrix_vector_crypto.h"
#include <filesystem>

using namespace std;
using namespace seal;

void compare(vec r, vec expected) {
	for (size_t i = 0; i < r.size(); ++i){
		// Test if value is within 0.1% of the actual value or 10 sig figs
		const auto difference = abs(r[i] - expected[i]);
		if (difference > max(0.000000001, 0.001 * abs(expected[i])))
		{
			cout << "\tERROR: difference of " << difference << " detected, where r[i]: " << r[i] << " and expected[i]: " << expected[i] << endl;
			//throw runtime_error("Comparison to expected failed");
		}		
	}
}

void decrypt_and_compare(const Ciphertext& ctxt_r, vec expected, Decryptor& decryptor, CKKSEncoder& encoder)
{	
		Plaintext ptxt_t;
		decryptor.decrypt(ctxt_r, ptxt_t);
		vec r;
		encoder.decode(ptxt_t, r);
		r.resize(expected.size());
		compare(r, expected);	
}

/// Copied from SEAL
uint64_t galois_elt_from_step(int step, size_t coeff_count)
{
	uint32_t n = util::safe_cast<uint32_t>(coeff_count);
	uint32_t m32 = util::mul_safe(n, uint32_t(2));
	uint64_t m = static_cast<uint64_t>(m32);

	if (step == 0)
	{
		return m - 1;
	}
	else
	{
		// Extract sign of steps. When steps is positive, the rotation
		// is to the left; when steps is negative, it is to the right.
		bool sign = step < 0;
		uint32_t pos_step = util::safe_cast<uint32_t>(abs(step));

		if (pos_step >= (n >> 1))
		{
			throw invalid_argument("step count too large");
		}

		pos_step &= m32 - 1;
		if (sign)
		{
			step = util::safe_cast<int>(n >> 1) - util::safe_cast<int>(pos_step);
		}
		else
		{
			step = util::safe_cast<int>(pos_step);
		}

		// Construct Galois element for row rotation
		uint64_t gen = 3;
		uint64_t galois_elt = 1;
		while (step--)
		{
			galois_elt *= gen;
			galois_elt &= m - 1;
		}
		return galois_elt;
	}
}

/// Copied from SEAL
vector<uint64_t> galois_elts_from_steps(shared_ptr<SEALContext> context_, const vector<int>& steps)
{
	vector<uint64_t> galois_elts;
	size_t coeff_count = context_->key_context_data()->parms().poly_modulus_degree();

	transform(steps.begin(), steps.end(), back_inserter(galois_elts),
		[&](auto s) { return galois_elt_from_step(s, coeff_count); });

	return galois_elts;
}

/// Create only the required power-of-two rotations
vector<uint64_t> galois_elts_custom(shared_ptr<SEALContext> context_, size_t dimension)
{
	vector<int> steps{};
	for (int i = 1; i <= dimension; i <<= 1)
	{
		steps.push_back(i);
		steps.push_back(-i);
	}
	return galois_elts_from_steps(context_, steps);
}

void example_rnn()
{
	/// dimension of word embeddings 
	const size_t embedding_size = 256;
	/// dimension of hidden thingy, MUST be the same as embedding size for square-ness of matrices
	const size_t hidden_size = embedding_size;
	/// Number of sentence chunks to process
	const size_t num_chunks = 7;
	
	// Setup Crypto
	timer t_setup;
	
	EncryptionParameters params(scheme_type::CKKS);
	vector<int> moduli = {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60};
	//TODO: Select proper moduli
	size_t poly_modulus_degree = 32768;
	double scale = pow(2.0, 40); //TODO: Select more appropriate scale

	params.set_poly_modulus_degree(poly_modulus_degree);
	params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, moduli));
	auto context = SEALContext::Create(params);
		
	KeyGenerator keygen(context);
	auto public_key = keygen.public_key();
	auto secret_key = keygen.secret_key();
	auto relin_keys = keygen.relin_keys();
	{
		ofstream fs("rnn.galk", ios::binary);
		keygen.galois_keys_save(galois_elts_custom(context, hidden_size), fs);
		//TODO: Generate only required galois keys		
	}

	Encryptor encryptor(context, public_key);
	encryptor.set_secret_key(secret_key);
	Decryptor decryptor(context, secret_key);
	CKKSEncoder encoder(context);
	
	print_parameters(context);
	cout << "Setup & generated keys in " << t_setup.get() << " ms." << endl;
	cout << "Galois Key Size: " << filesystem::file_size(filesystem::current_path() / "rnn.galk") << " Bytes" << endl;

	// Encrypt Input
	timer t_enc;
	/** 
	 *  In the case of translation, each word of the input sentence
	 *  is first mapped to a one-hot encoding of its index in a fixed dictionary.
	 *  These dictionaries tend to start at around 5000 words.
	 *  Chunks of the (one-hot encoded) sentence are than converted to embeddings in e.g. R^256
	 *  This is done using a simple pre-trained model.
	 *  These embeddings are now the inputs (x_0, x_1, ..) into the encryption
	 */
	
	/// Secret input, embeddings of sentence chunks into R^256
	vector<vec> x(num_chunks);
	for (size_t i = 0; i < x.size(); ++i)
	{
		x[i] = random_vector(embedding_size);
	}

	/// Secret input, batched into a single vector
	/// In order to preserve correctness in rotations, each x_i is duplicated
	/// i.e. x_batched = <br>
	///					 {x[0][0], x[0][1], x[0][2], ..., x[0][embedding_size],<br>
	///                   x[0][0], x[0][1], x[0][2], ..., x[0][embedding_size],<br>
	///                   x[1][0], x[1][1], x[1][2], ..., x[1][embedding_size],<br>
	///                   x[1][0], x[1][1], x[1][2], ..., x[1][embedding_size],<br>											
	///												....                       <br>
	///					  x[num_chunks][0], ..., x[num_chunks][embedding_size] <br>
	///					  x[num_chunks][0], ..., x[num_chunks][embedding_size]}
	vec x_batched(2 * num_chunks * embedding_size);
	for (size_t i = 0; i < x_batched.size(); ++i)
	{
		x_batched[i] = x[i / (2 * embedding_size)][i % embedding_size];
	}

	Plaintext ptxt_x;
	encoder.encode(x_batched, scale, ptxt_x);
	{
		ofstream fs("xs.ct", ios::binary);
		encryptor.encrypt_symmetric_save(ptxt_x, fs);
	}
	cout << "Encrypted input in " << t_enc.get() << " ms." << endl;	
	std::cout << "Ciphertext Size: " << filesystem::file_size(filesystem::current_path() / "xs.ct") << " Bytes" << endl;
		

	// Load Galois Keys
	timer t_load_glk;
	GaloisKeys galk;
	{
		ifstream fs("rnn.galk", ios::binary);
		galk.load(context, fs);
	}
	cout << "Loaded galois keys from disk in " << t_load_glk.get() << " ms." << endl;

	// Load Ciphertext
	timer t_load_ctxt;
	Ciphertext ctxt_x;
	{
		ifstream fs("xs.ct", ios::binary);
		ctxt_x.load(context, fs);
	}
	cout << "Loaded ciphertext of x from disk in " << t_load_ctxt.get() << " ms." << endl;

	// Create the Evaluator
	Evaluator evaluator(context);
	
	/**
	 *  The model parameters are split into the encoding phase parameters (M_x, M_h)
	 *  and the decoding phase parameters (TODO: Decoding-phase parameters documentation)
	 *  
	 *  During the encoding phase, we evaluate a simple RNN where the activation function
	 *  which would normally be ReLU is approximated by x^2
	 *  I.e. h_t = square(W_x * x_t + W_h * h_{t-1} + b)
	 *  Where W_x and W_h are weight matrices, b is a bias vector
	 *  x_t is the current input and h_{t-1} is the output from the previous layer
	 *  
	 */

	/// Encoding-phase weight matrix (part for x)
	auto M_x = random_square_matrix(hidden_size);
	
	/// Encoding-phase weight matrix (part for hidden input)
	auto M_h = random_square_matrix(hidden_size);

	/// Encoding-phase bias vector
	auto b = random_vector(hidden_size);

	/// Encoding-phase start value
	auto h0 = random_vector(hidden_size);

	/// Current hidden in/output 
	Ciphertext ctxt_h;

	// The very first cell is different, since h0 is still a plaintext:
	/// Time for 0th cell
	timer t_cell0;
	// Compute W_x * x_0
	ptxt_matrix_enc_vector_product_bsgs(galk, evaluator, encoder, hidden_size, diagonals(M_x),ctxt_x,ctxt_h);
	// Compute W_h * h_0 (still plaintext)
	Plaintext ptxt_t;
	auto t = mvp(M_h, h0);
	encoder.encode(duplicate(t), ctxt_h.parms_id(), ctxt_h.scale(), ptxt_t);
	// Add (W_x * x_0) + (W_h * h_0)
	evaluator.add_plain_inplace(ctxt_h, ptxt_t);
	// Add ((W_x * x_0) + (W_h * h_0)) + b
	Plaintext ptxt_b;
	encoder.encode(duplicate(b), ctxt_h.parms_id(), ctxt_h.scale(), ptxt_b);
	evaluator.add_plain_inplace(ctxt_h, ptxt_b);
	// Re-scale to avoid blow-up
	evaluator.rescale_to_next_inplace(ctxt_h);
	// Square
	evaluator.square_inplace(ctxt_h);	
	cout << "Computed 0th cell in " << t_cell0.get() << " ms." << endl;

	// Compare results with plaintext operation:
	vec h_expected = rnn_with_squaring(x[0], h0, M_x, M_h, b);
	decrypt_and_compare(ctxt_h, h_expected , decryptor, encoder);

	// Compute the next cells
	for(size_t i = 1; i < num_chunks; ++i)
	{
		/// Time for i-th cell
		timer t_cell;
		
		// Re-linearize h to prevent noise blow-up
		evaluator.relinearize_inplace(ctxt_h, relin_keys);
		// Re-scale h to prevent blow-up of plaintext bit size
		evaluator.rescale_to_next_inplace(ctxt_h);

		// rotate x to bring x[i] to the current position
		Ciphertext ctxt_x_rot;
		ctxt_x_rot = ctxt_x;
		evaluator.mod_switch_to_inplace(ctxt_x_rot, ctxt_h.parms_id());
		ctxt_x_rot.scale() = ctxt_h.scale(); // force scale to be exact
		evaluator.rotate_vector_inplace(ctxt_x_rot, i, galk);
		
		// Compute the RNN cell
		ptxt_weights_enc_input_rnn(galk, evaluator, encoder, hidden_size, diagonals(M_x), diagonals(M_h), b, ctxt_x_rot, ctxt_h);

		cout << "Computed " << i <<"th cell in " << t_cell.get() << " ms." << endl;

		// Compare results with plaintext operation:
		h_expected = rnn_with_squaring(x[i], h_expected, M_x, M_h, b);
		decrypt_and_compare(ctxt_h, h_expected, decryptor, encoder);
	}
	
	// TODO: Decoding-phase parameters

	// TODO: Decoding Phase
}
