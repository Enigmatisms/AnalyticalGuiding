/**
 * @file exp_test.cpp
 * 
 * @author Dinger
 * @note compilation: g++ ./exp_test.cpp -mavx2 -mfma
 * @copyright Copyright (c) 2023
 * 
 */

#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>
#include <immintrin.h>

union Double {
    double d;
    struct {
        int j, i;
    } n;
};

union Single {
    float f;
    int n;
};

class TicToc {
private:
    std::chrono::_V2::system_clock::time_point start;
public:
    void tic() {
        start = std::chrono::system_clock::now();
    }
    // return in milliseconds
    double toc() {
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / 1000.0;
    }
};


const double EXP_A = 1.0 * (1 << 20) / M_LN2;
const int EXP_C = 1023 * (1 << 20) - 60801;

const float EXP_Af = 1.0 * (1 << 23) / M_LN2;
const int EXP_Cf = 127 * (1 << 23) - 60801;

double fast_exp(double x) {
    Double xx;
    xx.n.i = EXP_A * x + EXP_C;
    return xx.d;
}

#define fastlog(x) (6 * (x - 1) / (x + 1 + 4 * sqrt(x)))

float fast_exp(float x) {
    Single xx;
    xx.n = EXP_Af * x + EXP_Cf;
    return xx.f;
}

float fast_exp2(float x) {
    x = 1.0 + x / (1 << 6);
    for (int i=0; i<6; i++)
        x *= x;
    return x;
}

float C1[] = {128*M_LN2, 64*M_LN2, 32*M_LN2, 16*M_LN2, 8*M_LN2, 4*M_LN2, 2*M_LN2, M_LN2};

float C2[] = {  0.405465108f, 0.223143551f, 0.117783036f, 0.060624622f,
                0.030771659f, 0.015504187f, 0.00778214f, 0.00389864f,
                0.00195122f, 0.000976086f, 0.000488162f, 0.000244111f};

float cordic_exp_pos(float x) {
    float y = 1.0;
    int i;
    for (i=0; i<8; i++) {
        if (x >= C1[i]) {
            x -= C1[i];
            y *= 1L <<(1<<(7-i));
        }
    }
    for (i=0; i<12; i++) {
        if (x >= C2[i]) {
            x -= C2[i];
            y += y / (1<<(i+1));
        }
    }
    return y;
}

float cordic_exp_neg(float x) {
    float y = 1.0;
    int i;
    x = -x;
    for (i=0; i<8; i++) {
        if (x >= C1[i]) {
            x -= C1[i];
            y /= 1L <<(1<<(7-i));
        }
    }
    for (i=0; i<12; i++) {
        if (x >= C2[i]) {
            x -= C2[i];
            uint32_t tmp = 1<<(i+1);
            y *= (float)tmp / (tmp + 1);
        }
    }
    return y;
}

inline float construct(uint8_t k) {
    Single s;
    s.n = ((127 - k) & 0xff) << 23;
    return s.f;
}

float my_exp(float x) {
    x = -x;
    uint k = floor(x / M_LN2);
    float r = x - k * M_LN2;
    float y = construct(k);
    for (int i=0; i<3; i++) {
        if (r >= C2[i]) {
            r -= C2[i];
            float tmp = construct(i+1);
            y /= 1.0 + tmp;
        }
    }
    r = 1.0 + r / (1 << 2);
    for (int i=0; i<2; i++)
        r *= r;
    return y / r;
}

float my_exp_fast(float x) {
    x = -x;
    float kf = x * 1.442695041;
    float k = ceil(kf);
    float r = (k - kf) * 0.6931471806;
    float y = construct(k);
    if (r >= 0.405465108f) {
        r -= 0.405465108f;
        y *= 1.5;
    }
    if (r >= 0.223143551f) {
        r -= 0.223143551f;
        y *= 1.25;
    }
    if (r >= 0.117783036f) {
        r -= 0.117783036f;
        y *= 1.125;
    }
    r = r / 4 + 1;
    r *= r;
    r *= r;
    return y * r;
}

#include <immintrin.h>

#define USE_FMA 0

/* compute exp(x) for x in [-87.33654f, 88.72283] 
   maximum relative error: 3.1575e-6 (USE_FMA = 0); 3.1533e-6 (USE_FMA = 1)
*/
__m256 faster_more_accurate_exp_avx2 (__m256 x)
{
    __m256 t, f, p, r;
    __m256i i, j;

    const __m256 l2e = _mm256_set1_ps (1.442695041f); /* log2(e) */
    const __m256 l2h = _mm256_set1_ps (-6.93145752e-1f); /* -log(2)_hi */
    const __m256 l2l = _mm256_set1_ps (-1.42860677e-6f); /* -log(2)_lo */
    /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
    const __m256 c0 =  _mm256_set1_ps (0.041944388f);
    const __m256 c1 =  _mm256_set1_ps (0.168006673f);
    const __m256 c2 =  _mm256_set1_ps (0.499999940f);
    const __m256 c3 =  _mm256_set1_ps (0.999956906f);
    const __m256 c4 =  _mm256_set1_ps (0.999999642f);

    /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
    t = _mm256_mul_ps (x, l2e);      /* t = log2(e) * x */
    r = _mm256_round_ps (t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); /* r = rint (t) */

#if USE_FMA
    f = _mm256_fmadd_ps (r, l2h, x); /* x - log(2)_hi * r */
    f = _mm256_fmadd_ps (r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */
#else // USE_FMA
    p = _mm256_mul_ps (r, l2h);      /* log(2)_hi * r */
    f = _mm256_add_ps (x, p);        /* x - log(2)_hi * r */
    p = _mm256_mul_ps (r, l2l);      /* log(2)_lo * r */
    f = _mm256_add_ps (f, p);        /* f = x - log(2)_hi * r - log(2)_lo * r */
#endif // USE_FMA

    i = _mm256_cvtps_epi32(t);       /* i = (int)rint(t) */

    /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
    p = c0;                          /* c0 */
#if USE_FMA
    p = _mm256_fmadd_ps (p, f, c1);  /* c0*f+c1 */
    p = _mm256_fmadd_ps (p, f, c2);  /* (c0*f+c1)*f+c2 */
    p = _mm256_fmadd_ps (p, f, c3);  /* ((c0*f+c1)*f+c2)*f+c3 */
    p = _mm256_fmadd_ps (p, f, c4);  /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
#else // USE_FMA
    p = _mm256_mul_ps (p, f);        /* c0*f */
    p = _mm256_add_ps (p, c1);       /* c0*f+c1 */
    p = _mm256_mul_ps (p, f);        /* (c0*f+c1)*f */
    p = _mm256_add_ps (p, c2);       /* (c0*f+c1)*f+c2 */
    p = _mm256_mul_ps (p, f);        /* ((c0*f+c1)*f+c2)*f */
    p = _mm256_add_ps (p, c3);       /* ((c0*f+c1)*f+c2)*f+c3 */
    p = _mm256_mul_ps (p, f);        /* (((c0*f+c1)*f+c2)*f+c3)*f */
    p = _mm256_add_ps (p, c4);       /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
#endif // USE_FMA

    /* exp(x) = 2^i * p */
    j = _mm256_slli_epi32 (i, 23); /* i << 23 */
    r = _mm256_castsi256_ps (_mm256_add_epi32 (j, _mm256_castps_si256 (p))); /* r = p * 2^i */

    return r;
}

int main() {
    float x = -1;
    float sum = 0;
    TicToc tt;
    tt.tic();
    for (int i=0; i<20000; i++) {
        sum += exp(x);
        x += 0.0001;
    }
    std::cout << "sum: " + std::to_string(sum) + ", exp: " + std::to_string(tt.toc()) + "ms\n";
    x = -1;sum = 0;
    tt.tic();
    for (int i=0; i<20000; i++) {
        sum += fast_exp(x);
        x += 0.0001;
    }
    std::cout << "sum: " + std::to_string(sum) + ", fast_exp: " + std::to_string(tt.toc()) + "ms\n";
    x = -1;sum = 0;
    tt.tic();
    for (int i=0; i<20000; i++) {
        sum += my_exp_fast(x);
        x += 0.0001;
    }
    std::cout << "sum: " + std::to_string(sum) + ", my_exp_fast: " + std::to_string(tt.toc()) + "ms\n";
    
    x = -1;sum = 0;
    tt.tic();
    __m256 sums = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
    __m256 xs = _mm256_setr_ps(-1, -1.0001, -1.0002, -1.0003, -1.0004, -1.0005, -1.0006, -1.0007);
    __m256 is = _mm256_setr_ps(0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008);
    for (int i=0; i<2500; i++) {
        sums = _mm256_add_ps(sums, faster_more_accurate_exp_avx2(xs));
        xs = _mm256_add_ps(xs, is);
    }
    float sumss[8];
    _mm256_storeu_ps(sumss, sums);
    for (int i=0; i<8; i++) {
        sum += sumss[i];
    }
    std::cout << "sum: " + std::to_string(sum) + ", avx: " + std::to_string(tt.toc()) + "ms\n";
    x = -1;sum = 0;
    tt.tic();
    for (int i=0; i<20000; i++) {
        sum += x;
        x += 0.0001;
    }
    std::cout << "sum: " + std::to_string(sum) + ", reference: " + std::to_string(tt.toc()) + "ms\n";

    x = -2;
    std::vector<float> ys(10000);
    std::vector<float> zs(10000);
    for (int i=0; i<10000; i++) {
        ys[i] = exp(x);
        zs[i] = my_exp_fast(x);
        x += 0.0004;
    }
    return 0;
}
