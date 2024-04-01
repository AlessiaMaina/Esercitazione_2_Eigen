#include <iostream>
#include <Eigen/Eigen>
#include <iomanip>

using namespace std;
using namespace Eigen;

VectorXd solutionWithPALU(const MatrixXd& A,
                          const VectorXd& b)
{
    VectorXd x_decPALU = A.fullPivLu().solve(b);              // In this case, fullPivLu it's the best PALU decomposition
    return x_decPALU;
}

VectorXd solutionWithQR(const MatrixXd& A,
                        const VectorXd& b)
{
    VectorXd x_decQR  = A.fullPivHouseholderQr().solve(b);    // In this case, fullPivHouseholderQr it's the best QR decomposition
    return x_decQR;
}

bool checkRelErrors(const MatrixXd& A,         // Represents a check for Relative Errors
                    const VectorXd& b,
                    const VectorXd exactSolution,
                    double& detA,
                    double& condA,
                    double& relativeErr_decPALU,
                    double& relativeErr_decQR)
{
    JacobiSVD<MatrixXd> svd(A);                // Check for Singular Values
    VectorXd singValA = svd.singularValues();
    condA = singValA.maxCoeff() / singValA.minCoeff();
    detA = A.determinant();

    if(singValA.minCoeff() < 1e-16)
    {
        relativeErr_decPALU = -1;
        relativeErr_decQR = -1;
        return false;
    }

    relativeErr_decPALU = (exactSolution - solutionWithPALU(A,b)).norm() / exactSolution.norm();
    relativeErr_decQR = (exactSolution - solutionWithQR(A,b)).norm() / exactSolution.norm();

    return true;
}


int main()
{
    // Definition of LINEAR_SYSTEM_1
    MatrixXd A1(2,2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    cout << scientific << setprecision(16) << "The matrix A1 is defined as: \n" << A1 << endl;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01,
        1.672384680188350e-01;
    cout << scientific << setprecision(16) << "The vector b1 is defined as: \n" << b1 << "\n" << endl;

    // Definition of LINEAR_SYSTEM_2
    MatrixXd A2(2,2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    cout << scientific << setprecision(16) << "The matrix A2 is defined as: \n" << A2  << endl;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04,
        4.259549612877223e-04;
    cout << scientific << setprecision(16) << "The vector b2 is defined as: \n" << b2 << "\n" << endl;

    // Definition of LINEAR_SYSTEM_3
    MatrixXd A3(2,2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    cout << scientific << setprecision(16) << "The matrix A3 is defined as: \n" << A3  << endl;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10,
        4.266924591433963e-10;
    cout << scientific << setprecision(16) << "The vector b3 is defined as: \n" << b3 << "\n" << endl;

    // Common solution of all three systems, called exactSolution = [-1.0e+0; -1.0e+00]
    VectorXd exactSolution = VectorXd::Constant(2, -1.0);


    // Results of LINEAR_SYSTEM_1
    cout << "LINEAR_SYSTEM_1" << endl;
    double detA1, condA1, relativeErr_decPALU_1, relativeErr_decQR_1;
    VectorXd x1_decPALU = solutionWithPALU(A1,b1);
    VectorXd x1_decQR = solutionWithQR(A1,b1);
    if(checkRelErrors(A1, b1, exactSolution, detA1, condA1, relativeErr_decPALU_1, relativeErr_decQR_1))
        cout << scientific << setprecision(16)
             << "DetA1: " << detA1 << "  -  RCondA1: " << 1.0 / condA1
             << "\n" << "The SOLUTION with PALU decomposition is: \n" << x1_decPALU
             << "\n" << "The RELATIVE ERROR with PALU decomposition is " << relativeErr_decPALU_1
             << "\n" << "The SOLUTION with QR decomposition is: \n" << x1_decQR
             << "\n" << "The RELATIVE ERROR with QR decomposition is " << relativeErr_decQR_1 << "\n" << endl;
    else
        cout << scientific << setprecision(16) << "The matrix A1 is singular! RcondA1: " << 1.0 / condA1 << endl;

    // Results of LINEAR_SYSTEM_2
    cout << "LINEAR_SYSTEM_2" << endl;
    double detA2, condA2, relativeErr_decPALU_2, relativeErr_decQR_2;
    VectorXd x2_decPALU = solutionWithPALU(A2,b2);
    VectorXd x2_decQR = solutionWithQR(A2,b2);
    if(checkRelErrors(A2, b2, exactSolution, detA2, condA2, relativeErr_decPALU_2, relativeErr_decQR_2))
        cout << scientific << setprecision(16)
             << "DetA2: " << detA2 << "  -  RCondA2: " << 1.0 / condA2
             << "\n" << "The SOLUTION with PALU decomposition is: \n" << x2_decPALU
             << "\n" << "The RELATIVE ERROR with PALU decomposition is " << relativeErr_decPALU_2
             << "\n" << "The SOLUTION with QR decomposition is: \n" << x2_decQR
             << "\n" << "The RELATIVE ERROR with QR decomposition is " << relativeErr_decQR_2 << "\n" << endl;
    else
        cout << scientific << setprecision(16) << "The matrix A2 is singular! RcondA2: " << 1.0 / condA2 << endl;

    // Results of LINEAR_SYSTEM_3
    cout << "LINEAR_SYSTEM_3" << endl;
    double detA3, condA3, relativeErr_decPALU_3, relativeErr_decQR_3;
    VectorXd x3_decPALU = solutionWithPALU(A3,b3);
    VectorXd x3_decQR = solutionWithQR(A3,b3);
    if(checkRelErrors(A3, b3, exactSolution, detA3, condA3, relativeErr_decPALU_3, relativeErr_decQR_3))
        cout << scientific << setprecision(16)
             << "DetA3: " << detA3 << "  -  RCondA3: " << 1.0 / condA3
             << "\n" << "The SOLUTION with PALU decomposition is: \n" << x3_decPALU
             << "\n" << "The RELATIVE ERROR with PALU decomposition is " << relativeErr_decPALU_3
             << "\n" << "The SOLUTION with QR decomposition is: \n" << x3_decQR
             << "\n" << "The RELATIVE ERROR with QR decomposition is " << relativeErr_decQR_3 << "\n" << endl;
    else
        cout << scientific << setprecision(16) << "The matrix A3 is singular! RcondA3: " << 1.0 / condA3 << endl;

    return 0;
}
