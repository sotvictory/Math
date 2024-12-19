#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

// g++ 1.cpp -o 1
// ./1

using Matrix = std::vector<std::vector<long double>>;
using Vector = std::vector<long double>;
using File = std::string;

const int exponent = 14;

Vector generateRandomVector(int n) {
    Vector x(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < n; ++i) {
        x[i] = dis(gen);
    }
    return x;
}

Matrix generateRandomMatrix(int n) {
    Matrix matrix(n, Vector(n));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

Matrix transpose(const Matrix matrix) {
    int n = matrix.size();
    Matrix result(n, Vector(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

Matrix multiply(const Matrix A, const Matrix B) {
    int n = A.size();
    Matrix result(n, Vector(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

Vector multiply(const Matrix& A, const Vector& x) {
    int n = A.size();
    Vector result(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }

    return result;
}

Vector multiplyR(const Matrix& R, const Vector& x) {
    int n = R.size();
    Vector result(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            result[i] += R[i][j] * x[j];
        }
    }

    return result;
}

Matrix subtract(const Matrix A, const Matrix B) {
    int n = A.size();
    Matrix result(n, Vector(n));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    
    return result;
}

Vector subtract(const Vector& a, const Vector& b) {
    int n = a.size();
    Vector result(n);

    for (int i = 0; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
    
    return result;
}

Vector backSubstitutionUpper(const Matrix& R, const Vector& b) {
    int n = R.size();
    Vector x(n);
    
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }
    
    return x;
}

void forwardSubstitutionLower(const Matrix& R, const Vector& b, Matrix& Q, int colIndex) {
    int n = R.size();

    for (int i = 0; i < n; ++i) {
        Q[i][colIndex] = b[i];
        for (int j = 0; j < i; ++j) {
            Q[i][colIndex] -= R[i][j] * Q[j][colIndex];
        }
        Q[i][colIndex] /= R[i][i];
    }
}

Matrix getQ(const Matrix& R, const Matrix& A) {
    int n = R.size();
    Matrix Q(n, Vector(n, 0.0));

    for (int j = 0; j < n; ++j) {
        Vector b(n);
        for (int i = 0; i < n; ++i) {
            b[i] = A[i][j];
        }

        forwardSubstitutionLower(R, b, Q, j);
    }

    return Q;
}

long double norm(const Vector& v) {
    long double sum = 0.0;
    for (long double val : v) {
        sum += val * val;
    }
    return sqrt(sum);
}

long double rmsNorm(const Vector& v) {
    long double sum = 0.0;
    for (long double val : v) {
        sum += val * val;
    }
    return sqrt(sum / v.size());
}

long double maximumNorm(const Matrix& matrix) {
    long double maxRowSum = 0.0;
    for (const auto& row : matrix) {
        long double rowSum = 0.0;
        for (const auto& val : row) {
            rowSum += std::abs(val);
        }
        maxRowSum = std::max(maxRowSum, rowSum);
    }
    return maxRowSum;
}

long double maximumNorm(const Vector& vector) {
    long double maxAbsValue = 0.0;
    for (const auto& val : vector) {
        maxAbsValue = std::max(maxAbsValue, std::abs(val));
    }
    return maxAbsValue;
}

void printMatrix(const Matrix matrix) {
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void printVector(const Vector& v) {
    for (const auto& val : v) {
        std::cout << val << " ";
    }
    std::cout << std::endl << std::endl;
}

void addIdentityToMatrix(Matrix& matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; ++i) {
        matrix[i][i] += 1;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////

bool checkEquality(const Matrix A, const Matrix Q, const Matrix R) {
    auto QR_product = multiply(Q, R);
    
    long double epsilon = 1e-10;
    
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            if (std::abs(A[i][j] - QR_product[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

bool isOrthogonal(const Matrix Q) {
    auto product = multiply(transpose(Q), Q);
    int n = product.size();

    long double epsilon = 1e-10;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j && std::abs(product[i][j] - 1.0) > epsilon) {
                return false;
            }
            if (i != j && std::abs(product[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

bool isUpperTriangular(const Matrix& R) {
    int n = R.size();

    long double epsilon = 1e-10;

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < std::min(i, n); ++j) {
            if (std::abs(R[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

void writeMatrixToCSV(const Matrix& matrix, const File& fileName) {
    std::ofstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + fileName);
    }

    for (const auto& row : matrix) {
        for (int j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

Matrix readMatrixFromCSV(const File& fileName) {
    Matrix matrix;
    std::ifstream file(fileName);
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для чтения: " + fileName);
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        Vector row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        matrix.push_back(row);
    }

    file.close();

    return matrix;
}

void validateQRDecomposition(const Matrix A, const Matrix Q, const Matrix R, const File& qfileName, const File& rfileName) {
    bool isEq = checkEquality(A, Q, R);
    bool isOrth = isOrthogonal(Q);
    bool isUpperTri = isUpperTriangular(R); 

    if (!isEq && !isOrth && !isUpperTri) {
        std::cout << "Проверка не пройдена: A != QR, матрица Q не является ортогональной и матрица R не является верхней треугольной." << std::endl;
    } else if (!isEq && !isOrth) {
        std::cout << "Проверка не пройдена: A != QR и матрица Q не является ортогональной." << std::endl;
    } else if (!isEq && !isUpperTri) {
        std::cout << "Проверка не пройдена: A != QR и матрица R не является верхней треугольной." << std::endl;
    } else if (!isOrth && !isUpperTri) {
        std::cout << "Проверка не пройдена: матрица Q не является ортогональной и матрица R не является верхне треугольной." << std::endl;
    } else if (!isEq) {
        std::cout << "Проверка не пройдена: A != QR." << std::endl;
    } else if (!isOrth) {
        std::cout << "Проверка не пройдена: матрица Q не является ортогональной." << std::endl;
    } else if (!isUpperTri) {
        std::cout << "Проверка не пройдена: матрица R не является верхней треугольной." << std::endl;
    } else {
        std::cout << "Проверка пройдена: A == QR, матрица Q является ортогональной и матрица R является верхней треугольной." << std::endl;
        try {
            writeMatrixToCSV(Q, qfileName);
            writeMatrixToCSV(R, rfileName);
            std::cout << "Матрицы Q и R успешно записаны в файлы." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Ошибка при записи в файл: " << e.what() << std::endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////

Matrix cholesky(const Matrix A) {
    int n = A.size();
    Matrix L(n, Vector(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            long double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum);
            } else {
                if (L[j][j] == 0) {
                    throw std::invalid_argument("Матрица не является положительно определенной.");
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    return L;
}

void choleskyQR(const Matrix& A, Matrix& Q, Matrix& R) {
    Matrix AtA = multiply(transpose(A), A);
    Matrix L = cholesky(AtA);

    R = transpose(L);
    Q = transpose(getQ(L, transpose(A)));
}

/////////////////////////////////////////////////////////////////////////////////////////////

Vector solveUsingQR(const Matrix& Q, const Matrix& R, const Vector& f) {
    Vector b(R.size(), 0.0);

    for (int j = 0; j < b.size(); ++j) {
        for (int i = 0; i < f.size(); ++i) {
            b[j] += Q[i][j] * f[i];
        }
    }

    return backSubstitutionUpper(R, b);
}

/////////////////////////////////////////////////////////////////////////////////////////////

void getLambda(const Matrix& A, long double *lambdaMin, long double *lambdaMax) {
    long double sum = 0;
    int n = A.size();

    for (int j = 1; j < n; ++j) {
        sum += std::abs(A[0][j]);
    }    

    *lambdaMax = sum + A[0][0];
    *lambdaMin = -sum + A[0][0];   

    for (int i = 1; i < n; ++i) {
        sum = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                sum += std::abs(A[i][j]);
            }
        }

        if ((sum + A[i][i]) > (*lambdaMax)) {
            *lambdaMax = sum + A[i][i];
        } else if ((-sum + A[i][i]) < (*lambdaMin)) {
            *lambdaMin = -sum + A[i][i];
        }
    }
}

Matrix Chebyshev(const Matrix& R, int k, const Vector& f, double lambdaMin, double lambdaMax) {
    int n = R.size(), m, parityCounter = 0;;
    Vector xPrev(n, 0.0), mi(k, 0.0), teta0(k, 0.0), teta1(k, 0.0), Rx(n, 0.0);
    Matrix solutions(k, Vector(n, 0.0));

    double tau_k, tau0 = 2.0 / (lambdaMax + lambdaMin);
    double ro0 = (lambdaMax - lambdaMin) / (lambdaMax + lambdaMin);
    teta0[0] = 1;

    for (m = 1; m < k; m *= 2) {
        Vector& currentTeta = (parityCounter % 2 == 0) ? teta1 : teta0;
        Vector& previousTeta = (parityCounter % 2 == 0) ? teta0 : teta1;

        for (int i = 1; i <= m; ++i) {
            currentTeta[2 * i - 2] = previousTeta[i - 1];
            currentTeta[2 * i - 1] = 4 * m - previousTeta[i - 1];
        }

        parityCounter++;
    } 

    Vector& currentTeta = (parityCounter % 2 == 1) ? teta1 : teta0;

    for (int i = 1; i <= k; ++i) {
        mi[i - 1] = cos(M_PI / (2 * k) * currentTeta[i - 1]);
    }

    for (int i = 0; i < k; ++i) {
        Rx = multiplyR(R, xPrev);
        tau_k = tau0 / (1 - ro0 * mi[i]);

        for (int j = 0; j < n; ++j) {
            solutions[i][j] = (f[j] - Rx[j]) * tau_k + xPrev[j];
            xPrev[j] = solutions[i][j];
        }
    }  

    return solutions;
}

/////////////////////////////////////////////////////////////////////////////////////////////

int main() {  
    const std::string SLAU_csv = "SLAU_var_8.csv";
    Matrix A, Q, R;

    try {
        A = readMatrixFromCSV(SLAU_csv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    
    addIdentityToMatrix(A);
    int n = A.size();

    // Оценка спектра матрицы

    long double lambdaMin = 0, lambdaMax = 0;
    getLambda(A, &lambdaMin, &lambdaMax);
    std::cout << "Границы спектра матрицы I + A: [" << lambdaMin << ";" << lambdaMax << "]" << std::endl;

    // Решение системы прямым методом

    choleskyQR(A, Q, R);

    Vector x = generateRandomVector(n);

    Vector f(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f[i] += A[i][j] * x[j];
        }
    }

    Vector solution = solveUsingQR(Q, R, f);

    auto solutionError = subtract(x, solution);
    long double normDifference = rmsNorm(solutionError);
    std::cout << "Норма погрешности решения с помощью прямого метода: " << normDifference << std::endl;

    // Решение системы итерационным методом

    f = multiply(transpose(Q), f);
    int maxIterations = pow(2, exponent), minIteration = -1, iterationThreshold = 1, exponentCounter = 1;
    std::ofstream plotData("plot_data.txt");

    Matrix solutions = Chebyshev(R, maxIterations, f, lambdaMin, lambdaMax);

    for (int i = 0; i < maxIterations; ++i) {
        Vector errorChebyshev = subtract(x, solutions[i]);
        double norm = rmsNorm(errorChebyshev);
        double relativeNorm = norm / rmsNorm(x);
        
        if (i + 1 == iterationThreshold) {
            std::cout << "Норма погрешности: " << std::scientific << norm << ", относительная погрешность: " << relativeNorm
                      << ", итераций: " << (i + 1) << ", показатель степени: " << (exponentCounter - 1) << std::endl;
            if ((minIteration == -1) && (normDifference > norm)) {
                minIteration = exponentCounter - 1;
            }

            iterationThreshold = pow(2, exponentCounter);
            ++exponentCounter;
        }
        
        plotData << std::scientific << norm << "," << i << std::endl;
    }
    
    plotData.close();

    if (minIteration == -1) {
        std::cout << "Необходимо рассмотреть большее количество итераций." << std::endl;
    } else {
        std::cout << "Наименьший показатель степени двойки, при котором погрешность решения не превосходит погрешность прямого метода: " << minIteration << std::endl;
    }

    return 0;
}