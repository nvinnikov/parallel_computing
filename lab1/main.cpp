#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <cmath>

#define TABLE_LENGTH 154
#define COLUMN_LENGTH 30


uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t) hi << 32) | lo;
}

std::string create_column(std::size_t column_length, const std::string &content) {
    std::size_t left_space_length = (column_length - content.length()) / 2;
    std::size_t right_space_length = column_length - content.length() - left_space_length;
    return std::string(left_space_length, ' ') + content + std::string(right_space_length, ' ');
}

void print_divider() {
    std::string divider = '|' + std::string(TABLE_LENGTH, '-') + '|';
    std::cout << divider << std::endl;
}

void print_header(const std::string &type) {
    print_divider();
    std::cout << '|' << create_column(TABLE_LENGTH, type) << '|' << std::endl;
    print_divider();
    std::cout << '|' << create_column(COLUMN_LENGTH, "OPERATIONS")
              << '|' << create_column(COLUMN_LENGTH, "TIME")
              << '|' << create_column(COLUMN_LENGTH, "OPERATIONS / SECOND")
              << '|' << create_column(COLUMN_LENGTH, "PROCESSOR CLOCKS")
              << '|' << create_column(COLUMN_LENGTH, "PROCESSOR CLOCKS / SECOND") << '|' << std::endl;
    print_divider();
}


void print_line(std::size_t operations, double seconds, double operations_per_second,
                uint64_t processor_clocks, double processor_clocks_per_second) {
    std::cout << '|' << create_column(COLUMN_LENGTH, std::to_string(operations))
              << '|' << create_column(COLUMN_LENGTH, std::to_string(seconds))
              << '|' << create_column(COLUMN_LENGTH, std::to_string(operations_per_second))
              << '|' << create_column(COLUMN_LENGTH, std::to_string(processor_clocks))
              << '|' << create_column(COLUMN_LENGTH, std::to_string(processor_clocks_per_second)) << '|' << std::endl;
}


int main() {
    {
        print_header("SIMPLE DIVISION");
        for (size_t i = 4; i < 9; ++i) {
            double val1 = 2.1;
            double val2 = 3.0;
            std::size_t operations = pow(10, i);
            auto start = std::chrono::system_clock::now();
            unsigned long long t1 = rdtsc();
            for (size_t j = 0; j < pow(10, i); ++j) {
                val1 = val1 / val2;
            }
            unsigned long long t2 = rdtsc();
            std::chrono::duration<double> duration = std::chrono::system_clock::now() - start;
            double seconds = duration.count();
            double operations_per_second = operations / seconds;
            uint64_t processor_clocks = t2 - t1;
            double processor_clocks_per_second = processor_clocks / seconds;
            print_line(operations, seconds, operations_per_second, processor_clocks, processor_clocks_per_second);
        }
        print_divider();
    }
    {
        print_header("SSE2 DIVISION");
        for (size_t i = 4; i < 9; ++i) {
            const double array[4] = {7.2, 7.2};
            const double array2[4] = {3.0, 3.0};
            __m128d val1 = _mm_load_pd(&array[0]);
            __m128d val2 = _mm_load_pd(&array2[0]);
            std::size_t operations = pow(10, i);
            auto start = std::chrono::system_clock::now();
            unsigned long long t1 = rdtsc();
            for (size_t j = 0; j < pow(10, i) / 2; ++j) {
                val1 = _mm_div_pd(val1, val2);
            }
            unsigned long long t2 = rdtsc();
            std::chrono::duration<double> duration = std::chrono::system_clock::now() - start;
            double seconds = duration.count();
            double operations_per_second = operations / seconds;
            uint64_t processor_clocks = t2 - t1;
            double processor_clocks_per_second = processor_clocks / seconds;
            print_line(operations, seconds, operations_per_second, processor_clocks, processor_clocks_per_second);
        }
        print_divider();
    }
    {
        print_header("AVX2 DIVISION");
        for (size_t i = 4; i < 9; ++i) {
            const double array[4] = {7.2, 7.2, 7.2, 7.2};
            const double array2[4] = {3.0, 3.0, 3.0, 3.0};
            __m256d val1 = _mm256_load_pd(&array[0]);
            __m256d val2 = _mm256_load_pd(&array2[0]);

            std::size_t operations = pow(10, i);
            auto start = std::chrono::system_clock::now();
            unsigned long long t1 = rdtsc();
            for (size_t j = 0; j < pow(10, i) / 4; ++j) {
                val1 = _mm256_div_pd(val1, val2);
            }
            unsigned long long t2 = rdtsc();
            std::chrono::duration<double> duration = std::chrono::system_clock::now() - start;
            double seconds = duration.count();
            double operations_per_second = operations / seconds;
            uint64_t processor_clocks = t2 - t1;
            double processor_clocks_per_second = processor_clocks / seconds;
            print_line(operations, seconds, operations_per_second, processor_clocks, processor_clocks_per_second);
        }
        print_divider();
    }
}

