#include "example_rnn.h"
#include "helpers.h"
#include "timer.h"
#include "matrix_vector_crypto.h"
#include <filesystem>

using namespace std;
using namespace seal;

void example_rnn()
{
	/// dimension of word embeddings 
	const size_t embedding_size = 256;
	/// dimension of hidden thingy, same as embedding size for square-ness of matrices
	const size_t hidden_size = embedding_size;
	/// Number of sentence chunks to process
	const size_t num_chunks = 7;
	
	// Setup Crypto
	timer t_setup;
	
	EncryptionParameters params(scheme_type::CKKS);
	vector<int> moduli = {50, 40, 40, 40, 40, 40, 40, 40, 40, 59}; //TODO: Select proper moduli
	size_t poly_modulus_degree = 16384; // TODO: Select appropriate degree
	double scale = pow(2.0, 40); //TODO: Select appropriate scale

	params.set_poly_modulus_degree(poly_modulus_degree);
	params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, moduli));
	auto context = SEALContext::Create(params);
		
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
	cout << "Encrypted input in" << t_enc.get() << " ms." << endl;	
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

	
	//TODO: Lookup computation again!!!
	/**
	 *  The model parameters are split into the encoding phase parameters (M_x, M_h)
	 *  and the decoding phase parameters (b, W,U,V,c)
	 *  
	 *  During the encoding phase, we want to compute the following plaintext operation:
	 *  Given a start vector s = [s_x | s_h] with s_x in R^embedding_size, s_h in R^hidden_size
	 *  and the weight matrix M = [M_x | M_h] with M_x of size embedding_size x hidden_size and
	 *  M_h of 
	 *  We compute v_0 = ....
	 *  
	 */

	/// Encoding-phase weight matrix (part for x)
	auto M_x = random_square_matrix(embedding_size);
	
	/// Encoding-phase weight matrix (part for hidden input)
	auto M_h = random_square_matrix(embedding_size);


	// Weight vectors/matrices for decoding phase 
	auto b = random_vector(embedding_size);
	auto W = random_square_matrix(embedding_size);
	auto U = random_square_matrix(embedding_size);
	auto V = random_square_matrix(embedding_size);
	auto c = random_vector(embedding_size);


	// Start thingies?
	auto s_x = random_vector(embedding_size);
	print_vector(s_x, "s_x:");
	auto s_h = random_vector(embedding_size);
	print_vector(s_h, "s_h:");

	// Create the Evaluator
	Evaluator evaluator(context);
	
	// Encode diagonals as ptxts
	cout << "Encoding plaintext model...";
	// TODO: Do this at appropriate scale when required
	vector<Plaintext>
		ptxt_diagonals_W_x(embedding_size),
		ptxt_diagonals_W_h(embedding_size),
		ptxt_diagonals_W(embedding_size),
		ptxt_diagonals_U(embedding_size),
		ptxt_diagonals_V(embedding_size);
	for (size_t i = 0; i < embedding_size; ++i)
	{
		encoder.encode(duplicate(diag(M_x,i)), scale, ptxt_diagonals_W_x[i]);
		encoder.encode(duplicate(diag(M_h, i)), scale, ptxt_diagonals_W_h[i]);
		encoder.encode(duplicate(diag(W, i)), scale, ptxt_diagonals_W[i]);
		encoder.encode(duplicate(diag(U,i)), scale, ptxt_diagonals_U[i]);
		encoder.encode(duplicate(diag(V,i)), scale, ptxt_diagonals_V[i]);
	}
	cout << "...done" << endl;

	// Encoding Phase
	cout << "Starting encoding phase of RNN:" << endl;

	// Compute W_x * x_1 for the first block
	cout << "Computing W_x * x_1...";
	Ciphertext h1_ctxt;
	ptxt_matrix_enc_vector_product(galk, evaluator, embedding_size, ptxt_diagonals_W_x, ctxt_x, h1_ctxt);
	cout << "...done" << endl;

	// Compute encrypted result:
	{
		Plaintext ptxt_block1;
		decryptor.decrypt(h1_ctxt, ptxt_block1);
		vec block1;
		encoder.decode(ptxt_block1, block1);
		cout << "Encrypted result:" << endl;
		print_vector(vector(block1.begin(), block1.begin() + embedding_size));
	}

	// Compute expected result:
	{
		vec x1 = vec(x_batched.begin(), x_batched.begin() + embedding_size);
		vec r = mvp(M_x, x1);
		cout << "Expected result: " << endl;
		print_vector(r);
	}

	// Compute W_h * h_0 and add to previous computed W_x * x_1
	cout << "Compute (ptxt) W_h * h_0 and add to previously computed (W_x * x_1)...";
	auto h_0 = add(mvp(M_x, s_x), mvp(M_h, s_h));
	auto rhs_1 = mvp(M_h, h_0);
	Plaintext ptxt_rhs1;
	encoder.encode(rhs_1, h1_ctxt.scale(), ptxt_rhs1);
	evaluator.add_plain_inplace(h1_ctxt, ptxt_rhs1);
	cout << "...done" << endl;

	// Compute expected result:
	{
		print_vector(h_0, "h_0 = W_x * s_x:");
		print_vector(rhs_1, "rhs_1 = W_h * h_0:");
		vec x1 = vec(x_batched.begin(), x_batched.begin() + embedding_size);
		vec r = add(rhs_1, mvp(M_x, x1));
		cout << "Expected result: " << endl;
		print_vector(vector(r.begin(), r.begin() + embedding_size));
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
	for (size_t i = 2; i <= num_chunks; ++i)
	{
		// Compute W_h * h_(i-1)
		cout << "Compute W_h * h_" << i - 1 << "...";
		ptxt_matrix_enc_vector_product(galk, evaluator, embedding_size, ptxt_diagonals_W_h, h, tmp_whh);
		cout << "...done" << endl;

		// Compute W_x * x_i
		cout << "Compute W_x * x_" << i;
		evaluator.rotate_vector(ctxt_x, 2 * embedding_size, galk, xs_rot); //TODO: w_x * x_i from batching
		ptxt_matrix_enc_vector_product(galk, evaluator, embedding_size, ptxt_diagonals_W_x, xs_rot, tmp_wxx);
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

	// Expected Result
	cout << "Expected result of encoding phase: TBD" << endl;
	vec h_expected;
	// Encrypted Result
	{
		Plaintext ptxt_h;
		decryptor.decrypt(h, ptxt_h);
		encoder.decode(ptxt_h, h_expected);
		cout << "Encrypted result of encoding phase:" << endl;
		print_vector(h_expected);
	}


	// TODO: Decoding Phase
	cout << "Starting decoding phase of the RNN:" << endl;
}
