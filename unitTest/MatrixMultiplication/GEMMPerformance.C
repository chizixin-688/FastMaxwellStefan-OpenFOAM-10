#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mkl.h>
#include "myGEMM.h"
#include <mkl_cblas.h>
#include <iomanip>
#include <mkl_service.h>
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
    const std::vector<double>& C_ref,  // MKL结果
    const std::vector<double>& C_test, // 你的GEMM结果
    int M, 
    int N
)
{
    double l2_error = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double ref_norm = 0.0;

    const double eps = 1e-16; // 避免除零

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

// ======================
// 测试主函数
// ======================
void test_case(int M, int repeat)
{
    std::cout << "\nMatrix size: "
              << M << " x " << M
              << " * " << M << " x " << M << std::endl;

    std::vector<double> A(M*M);
    std::vector<double> B(M*M);
    std::vector<double> C1(M*M);
    std::vector<double> C2(M*M);

    fill_random(A);
    fill_random(B);

    // MKL GEMM
    double time_mkl = benchmark([&]() {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, M, M,
                    1.0,
                    A.data(), M,
                    B.data(), M,
                    0.0,
                    C1.data(), M);
    },repeat);

    // 自定义 GEMM
    double time_my=0;
    if(M%8==0)
    {
        time_my = benchmark
        (
            [&]()
            {
                MatrixMultiply484_0(C2.data(), A.data(), B.data(), M);
            },
            repeat
        );
    }
    else if(M%8==1)
    {
        time_my = benchmark
        (
            [&]()
            {
                MatrixMultiply484_1(C2.data(), A.data(), B.data(), M);
            },
            repeat
        );
    }
    else if(M%8==2)
    {
        time_my = benchmark
        (
            [&]()
            {
                MatrixMultiply484_2(C2.data(), A.data(), B.data(), M);
            },
            repeat
        );
    }
    else if(M%8==3)
    {
        time_my = benchmark
        (
            [&]()
            {
                MatrixMultiply484_3(C2.data(), A.data(), B.data(), M);
            },
            repeat
        );
    }
    else if(M%8==4)
    {
        time_my = benchmark
        (
            [&]()
            {
                MatrixMultiply484_4(C2.data(), A.data(), B.data(), M);
            },
            repeat
        );
    }
    else if(M%8==5)
    {
        time_my = benchmark
        (
            [&]()
            {
                MatrixMultiply484_5(C2.data(), A.data(), B.data(), M);
            },
            repeat
        );
    }
    else if(M%8==6)
    {
        time_my = benchmark
        (
            [&]()
            {
                MatrixMultiply484_6(C2.data(), A.data(), B.data(), M);
            },
            repeat
        );
    }
    else if(M%8==7)
    {
        time_my = benchmark
        (
            [&]()
            {
                MatrixMultiply484_7(C2.data(), A.data(), B.data(), M);
            },
            repeat
        );
    }

    std::cout << "MKL dgemm time: " << time_mkl << " s\n";
    std::cout << "My  gemm time: " << time_my  << " s\n";
    std::cout << "Speedup (MKL / My): " << time_my / time_mkl << "\n";
    compute_error_metrics(C1,C2,M,M);
}

// ======================
// main
// ======================
int main()
{
    mkl_set_cpu_arch(MKL_CPU_ARCH_AVX2);
    mkl_set_num_threads(1);
    mkl_set_dynamic(0);
    test_case(8192,1);
    test_case(4096,1); //>48 second
    test_case(2048,1); //6 second
    test_case(1024,100);//44 second
    test_case(512, 1000);//45 second
    test_case(256, 10000);//65 second

    test_case(128, 2e5);

    test_case(64,2e6); //62 second

    test_case(32,2e7); // 70 second
    test_case(16,1e8); // 80 second

    return 0;
}
