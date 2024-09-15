
// eigen_computation.cpp
#include <emscripten/bind.h>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

std::vector<std::complex<double>> computeEigenvalues(const std::vector<std::vector<double>>& matrix) {
    int n = matrix.size();
    MatrixXd mat(n, n);
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            mat(i, j) = matrix[i][j];
    
    Eigen::EigenSolver<MatrixXd> solver(mat);
    std::vector<std::complex<double>> eigenvalues;
    for(int i = 0; i < solver.eigenvalues().size(); ++i)
        eigenvalues.push_back(solver.eigenvalues()[i]);
    return eigenvalues;
}

EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::register_vector<std::vector<double>>("Matrix");
    emscripten::register_vector<std::complex<double>>("Eigenvalues");
    emscripten::function("computeEigenvalues", &computeEigenvalues);
}
