#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mkl.h>
#include <mkl_cblas.h>
#include "LUsolver.H"
#include <iomanip>
#include <cstring>

//#include "fvCFD.H"
//#include "LUscalarMatrix.H"
//#include "scalarSquareMatrix.H"
// ======================
// 工具函数
// ======================
void fill_random(std::vector<double>& mat)
{
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto& v : mat)
    {
        v = dist(gen);
    }
}

// 计时函数
template<typename Func>
double benchmark(Func f, int repeat = 10)
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat; ++i)
        f();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    return diff.count() / repeat;
}


void compute_error_metrics
(
    const std::vector<double>& C_ref,  
    const std::vector<double>& C_test, 
    int M, 
    int N
)
{
    double l2_error = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double ref_norm = 0.0;

    const double eps = 1e-16; 

    for(int i = 0; i < M*N; ++i)
    {
        double diff = C_test[i] - C_ref[i];
        l2_error += diff * diff;
        max_abs_error = std::max(max_abs_error, std::abs(diff));
        max_rel_error = std::max(max_rel_error, std::abs(diff) / (std::abs(C_ref[i]) + eps));
        ref_norm += C_ref[i] * C_ref[i];
    }

    l2_error = std::sqrt(l2_error);
    ref_norm = std::sqrt(ref_norm);
    double rel_l2 = l2_error / (ref_norm + eps);

    //std::cout.precision(17);
    std::cout <<std::setprecision(17)<< "L2 error:         " << l2_error << "\n";
    std::cout <<std::setprecision(17)<< "Relative L2 error:" << rel_l2 << "\n";
    std::cout <<std::setprecision(17)<< "Max absolute error:" << max_abs_error << "\n";
    std::cout <<std::setprecision(17)<< "Max relative error:" << max_rel_error << "\n";
}
double rand_double(double min, double max) 
{
    return min + (max - min) * (rand() / double(RAND_MAX));
}

int test_case(int M, int repeat)
{
    Foam::LUsolver Lu(M);
    {
        int remain = M%4;
        int align = M - remain;
        std::vector<int> coins = {56,52,48,44,40,36,32,28,24,20,16,12,8,4};
        for (int i = 0; i < 14; i++) 
        { 
            int n = align / coins[i]; 
            Lu.callTable[i] = n; 
            align = align - n*coins[i]; 
        }
    }

    std::vector<double> A(M*M);
    for (int i = 0; i < M; ++i) 
    {
        for (int j = 0; j < M; ++j) 
        {
            if (i == j) 
            {
                // 对角元 100-1000
                A[i*M + j] = rand_double(100.0, 1000.0);
            } 
            else 
            {
                // 非对角元 0-1
                A[i*M + j] = rand_double(0.0, 1.0);
            }
        }
    }    

    std::vector<double> A_backup = A;
    std::vector<double> A_mkl = A;

    std::vector<double> invA(M*M);

// ----------------- MKL LAPACKE -----------------
auto t1 = std::chrono::high_resolution_clock::now();
for(int i=0;i<repeat;i++)
{
    std::copy(A_backup.begin(), A_backup.end(), A_mkl.begin());
    std::vector<int> ipiv(M);
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, M, M, A_mkl.data(), M, ipiv.data());
    if (info != 0) { std::cout << "MKL LU failed\n"; return -1; }
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, M, A_mkl.data(), M, ipiv.data());
}
auto t2 = std::chrono::high_resolution_clock::now();
double t_mkl = std::chrono::duration<double>(t2 - t1).count();


// ----------------- My invA -----------------
auto t3 = std::chrono::high_resolution_clock::now();
for(int i=0;i<repeat;i++)
{
    std::copy(A_backup.begin(), A_backup.end(), A.begin());
    Lu.ReAssign(A.data());
    Lu.Block4LUDecompose();
    std::memset(invA.data(),0,M*M*sizeof(double));
    Lu.setInvMatrix(invA.data());
    Lu.calcInvMatrix(invA.data());
}
auto t4 = std::chrono::high_resolution_clock::now();
double t_myInv = std::chrono::duration<double>(t4 - t3).count();


// ----------------- Output -----------------
    std::cout << "Matrix size: " << M << " s\n";
    std::cout << "MKL dgemm time: " << t_mkl/repeat << " s\n";
    std::cout << "My  gemm time: " << t_myInv/repeat << " s\n";
    std::cout << "Speedup (MKL / My): " << t_myInv / t_mkl << "\n";

    compute_error_metrics(A_mkl,invA,M,M);
    return 1;
}



// ======================
// main
// ======================
int main()
{

    mkl_set_num_threads(1);
    mkl_set_dynamic(0);
    test_case(16,100000);
    test_case(32,100000);
    test_case(64,100000);
    test_case(128,1000);
    test_case(256,100);
    return 0;
}
