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

long double norm(const Vector& v) {
    long double sum = 0.0;
    for (long double val : v) {
        sum += val * val;
    }
    return sqrt(sum);
}

Vector computeReflectionVector(const Vector& y) {
    long double yNorm = norm(y);
    Vector w(y.size());

    w = y;
    long double alpha = (y[0] >= 0 ? 1 : -1) * yNorm;
    w[0] += alpha;

    long double vNorm = norm(w);
    for (long double& val : w) {
        val /= vNorm;
    }

    return w;
}

void householderQR(const Matrix& A, Matrix& Q, Matrix& R) {
    int n = A.size();
    R = A;

    Q.assign(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) {
        Q[i][i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        Vector y(n - k);
        for (int i = k; i < n; ++i) {
            y[i - k] = R[i][k];
        }

        Vector w = computeReflectionVector(y);
        Vector dotProduct(n - k);

        for (int j = k; j < n; ++j) {
            long double dotProd = 0.0;
            for (int i = k; i < n; ++i) {
                dotProd += w[i - k] * R[i][j];
            }
            dotProduct[j - k] = dotProd;
        }

        for (int i = k; i < n; ++i) {
            for (int j = k; j < n; ++j) {
                R[i][j] -= 2 * w[i - k] * dotProduct[j - k];
            }
        }

        for (int i = 0; i < n; ++i) {
            long double tmp = 0.0;
            for (int j = k; j < n; ++j) {
                tmp += w[j - k] * Q[i][j];
            }
            for (int j = k; j < n; ++j) {
                Q[i][j] -= 2 * w[j - k] * tmp;
            }
        }
    }
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

int main() {  
    const std::string SLAU_csv = "SLAU_var_8.csv";
    Matrix A, Q, R;

    //A = generateRandomMatrix(30);

    try {
        A = readMatrixFromCSV(SLAU_csv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    // Холецкий

    std::cout << "Холецкий" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    choleskyQR(A, Q, R);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ms = end_time - start_time;

    //validateQRDecomposition(A, Q, R, "Q_Cholesky.csv", "R_Cholesky.csv");
    auto D = subtract(A, multiply(Q, R));
    long double normDifference = maximumNorm(D);
    std::cout << "Матричная норма разности ||A - QR||: " << normDifference << std::endl;
    std::cout << "Время на построение разложения: " <<  duration_ms.count() * 1000 << " мс" << std::endl << std::endl; 

    // Решение системы

    int n = A[0].size();
    Vector x = generateRandomVector(n);
    Vector f(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f[i] += A[i][j] * x[j];
        }
    }

    start_time = std::chrono::high_resolution_clock::now();

    Vector solution = solveUsingQR(Q, R, f);

    end_time = std::chrono::high_resolution_clock::now();
    duration_ms = end_time - start_time;
    std::cout << "Время, потраченное на решение системы: " << duration_ms.count() * 1000 << " мс" << std::endl; 

    //std::cout << "\nСгенерированный вектор x:" << std::endl;
    //printVector(x);
    
    //std::cout << "Вектор f:" << std::endl;
    //printVector(f);

    //std::cout << "Решение системы Ax = f:" << std::endl;
    //printVector(solution);

    Vector Ax = multiply(A, solution);
    //std::cout << "Ax:" << std::endl;
    //printVector(Ax);

    Vector r = subtract(f, Ax);
    double maxNormValue = maximumNorm(r);

    //std::cout << "Невязка r:" << std::endl;
    //printVector(r);
    
    std::cout << "Максимум-норма невязки: " << maxNormValue << std::endl;

    Vector delta = subtract(solution, x);
    double maxNormError = maximumNorm(delta);

    //std::cout << "Погрешность решения:" << std::endl;
    //printVector(delta);

    std::cout << "Максимум-норма погрешности: " << maxNormError << std::endl << std::endl;

    // Хаусхолдер

    std::cout << "Хаусхолдер" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();

    householderQR(A, Q, R);

    end_time = std::chrono::high_resolution_clock::now();
    duration_ms = end_time - start_time;

    //validateQRDecomposition(A, Q, R, "Q_Householder.csv", "R_Householder.csv");
    D = subtract(A, multiply(Q, R));
    normDifference = maximumNorm(D);
    std::cout << "Матричная норма разности ||A - QR||: " << normDifference << std::endl;
    std::cout << "Время на построение разложения: " << duration_ms.count() * 1000 << " мс" << std::endl;

    return 0;
}