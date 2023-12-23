#ifndef CAMERAREADYCODE_LEAFSCORE_H
#define CAMERAREADYCODE_LEAFSCORE_H

#include <bitset>
#include <vector>

class LeafScore{
public:
    int K;  // no. of candidate parents

    LeafScore(int K, std::vector<double>& score_array);
    //~LeafScore();
    void precompute_sum_and_max();
    double sum(std::vector<int>& A, std::vector<int>& A_prime);
    std::pair<double, std::vector<int> > max(std::vector<int>& A, std::vector<int>& A_prime);

private:
    std::vector<double> score_array;  // scores p_(G_i) for each parent set G_i
    std::vector<double> f;
    std::vector<double> f_max;
    std::vector<long long> f_max_idx;
};


#endif


