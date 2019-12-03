#include "example_ckks.h"
#include "polynomial_evaluation.h"
#include "mvproduct.h"

using namespace std;
int main()
{
#ifdef SEAL_VERSION
	cout << "Microsoft SEAL version: " << SEAL_VERSION << endl;
#endif

	while (true) {
		cout << endl;
		cout << "Select an example to run:" << endl;
		cout << "  1. Kim's optimized CKKS example" << endl;
		cout << "  2. Polynomial Evaluation" << endl;
		cout << "  3. Matrix-Vector Product" << endl;
		cout << "  0. Quit" << endl;

		int selection = 0;
		if (!(cin >> selection))
		{
			cout << "Invalid option." << endl;
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			continue;
		}

		switch (selection)
		{
		case 1:
			bootcamp_demo();
			break;

		case 2:
			example_polyeval();
			break;

		case 3:
			example_mvproduct();
			break;
		case 0:
			cout << endl;
			return 0;

		default:
			cout << "Invalid option." << endl;
		}
	}

	return 0;
}