#include "LeafScore.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <bitset>

std::string int_to_string(int n, int str_length, int base) {
    std::string int_to_digit = "0123456789";
    std::string str(str_length, '0');
    int s_idx = str_length - 1;
    while (n > 0) {
        str[s_idx] = int_to_digit[n%base];
        n /= base;
        s_idx--;
    }
    return str;
}

//vector<int> pow_array(int a, int b) { vector<int> res; for(int n = 1; res.size() < b; n *= a) res.push_back(n) ; return res};

LeafScore::LeafScore(int K, std::vector<double> &score_array) {
    this->K = K;
    this->score_array = score_array;
    this->f = std::vector<double> (std::pow(3, K));
    this->f_max = std::vector<double> (std::pow(3, K));
    this->f_max_idx = std::vector<long long> (std::pow(3, K));
}

void LeafScore::precompute_sum_and_max() {
    // 1) precompute sum and max. 0 = unspecified, 1 = A_i, 2 = A_i'

    int max_pset_idx = 1 << this->K;
    // every bit which flips from A_i to A_i'


    // Initialize scores
    for (int pset_idx = 0; pset_idx < max_pset_idx; pset_idx++) {

        std::string f_idx_str(this->K, '2');
        auto pset_idx_bs = std::bitset<32>(pset_idx);
        for (int i=0; i<this->K; i++) {f_idx_str[this->K-1-i] -= pset_idx_bs[i];};

        int f_idx = std::stoi(f_idx_str,nullptr,3);

        f[f_idx] = this->score_array[pset_idx];
        f_max[f_idx] = this->score_array[pset_idx];
        f_max_idx[f_idx] = pset_idx;
    }
    //std::cout << std::endl;
    for (int f_idx = pow(3,this->K) - 1; f_idx>=0; f_idx--) {
        std::string f_idx_str = int_to_string(f_idx,this->K,3);
        //std::cout << f_idx_str << std::endl;
        // if no element is 0, f[f_idx] is not altered
//for (int i=f_idx_str.size()-1; i>=f_idx_str.size() - this->K; --it)

        for (auto &ch: f_idx_str) {
            //std::cout << "DDY" << std::endl;
            if (ch == '0') {
                //std::cout << f_idx_str << " ";

                ch = '1';
                double f_val_1 = f[std::stoi(f_idx_str,nullptr,3)];
                double f_max_val_1 = f_max[std::stoi(f_idx_str,nullptr,3)];
                double f_max_p_idx_1 = f_max_idx[std::stoi(f_idx_str,nullptr,3)];
                ch = '2';
                double f_val_2 = f[std::stoi(f_idx_str,nullptr,3)];
                double f_max_val_2 = f_max[std::stoi(f_idx_str,nullptr,3)];
                double f_max_p_idx_2 = f_max_idx[std::stoi(f_idx_str,nullptr,3)];

                double a = std::max(f_val_1, f_val_2);
                f[f_idx] = a + log(exp(f_val_1 - a) + exp(f_val_2 - a)); // logaddexp
//                std::cout << i << " " << f_idx << std::endl;
               // std::cout << "DK " << f[f_idx] << std::endl;
                if (f_val_1 > f_val_2) {
                    f_max[f_idx] = f_max_val_1;
                    f_max_idx[f_idx] = f_max_p_idx_1;
                }
                else {
                    f_max[f_idx] = f_max_val_2;
                    f_max_idx[f_idx] = f_max_p_idx_2;
                }
//                std::cout << f_idx_str << " ";
//                f_idx_str[i] = '2'; f[f_idx] += f[std::stoi(f_idx_str,nullptr,3)];
//                std::cout << f_idx_str << " ";
//                std::cout << i << std::endl;
                break;
            }
        }
    }

    // iterate in reverse order from 3^K. find b (First position with 0). work out value by adding the appropriate values.


}

double LeafScore::sum(std::vector<int>& A, std::vector<int>& A_prime) {
    int f_idx = 0;
    for (int i = 0; i < A.size(); i++) {
        f_idx += std::pow(3, A[i]);
    }
    for (int i = 0; i < A_prime.size(); i++) {
        f_idx += 2 * std::pow(3, A_prime[i]);
    }
    return f[f_idx];
}

std::pair<double, std::vector<int> > LeafScore::max(std::vector<int>& A, std::vector<int>& A_prime) {
    int f_idx = 0;
    for (int i = 0; i < A.size(); i++) {
        f_idx += std::pow(3, A[i]);
    }
    for (int i = 0; i < A_prime.size(); i++) {
        f_idx += 2 * std::pow(3, A_prime[i]);
    }
    std::vector<int> max_pset_elements;
    std::string pset_idx_string = int_to_string(f_max_idx[f_idx], this->K, 2);
    for (int i = 0; i < this->K; i++) {
        if (pset_idx_string[this->K - 1 - i] == '1')
            max_pset_elements.push_back(i);
    }
    return std::make_pair(f_max[f_idx], max_pset_elements);
}


int main() {
    std::vector<double> score_array(std::pow(2, 2));
    score_array = {0.3, 0.6, 0.7, 0.9};
    LeafScore g = LeafScore(2, score_array);
    g.precompute_sum_and_max();
}



#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(LeafScore, m) {
py::class_<LeafScore>(m, "LeafScore")
.def(py::init<int, std::vector<double> &>())
.def("precompute_sum_and_max", &LeafScore::precompute_sum_and_max)
.def("sum", &LeafScore::sum)
.def("max", &LeafScore::max);
}





// More "efficient?" but also difficult implementation
//void LeafScore::precompute_sum_and_max() {
//    // 1) precompute sum and max. 0 = unspecified, 1 = A_i, 2 = A_i'
//
//    int max_pset_idx = 1 << this->K;
//    // every bit which flips from A_i to A_i'
//    //vector<int> powers_of_three = pow_array(3, K);
//    std::vector<int> increments;
//    for (int n = 1; increments.size() < this->K; n *= 3) {increments.push_back(n/2 + 1); std::cout << n/2 + 1 << std::endl;};
//    std::cout << std::endl;
//    int f_idx = pow(3, K) - 1;
//    for (int pset_idx = 0; pset_idx < max_pset_idx; pset_idx++) {
//        // Initialize scores
//        f[f_idx] = this->score_array[pset_idx];
//
//        // Work out next index
//        int flipped_bit = ~pset_idx & (pset_idx+1);
//        std::cout << "FB " << flipped_bit << std::endl;
//        f_idx -= increments[flipped_bit];
//        std::cout << "FI " << f_idx << std::endl;
//    }
//
//    // iterate in reverse order from 3^K. find b (First position with 0). work out value by adding the appropriate values.
//
//
//}