// -*- flycheck-clang-language-standard: "c++17"; -*-

#include <fstream>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>
#include "../bitmap/bitmap.hpp"
#include "common.hpp"
#include "CandidateRestrictedScore.hpp"

using std::pair;
using std::cout;
using std::endl;
using std::ofstream;
using std::ios_base;


CandidateRestrictedScore::CandidateRestrictedScore(double* score_array,
												   int* C, int n, int K,
												   int cc_limit, double cc_tol,
												   double isum_tol,
												   string logfile,
												   bool silent) {

  coutbuf = std::cout.rdbuf();
  // Second argument is "append".
  ofstream out(logfile, ios_base::app);
  if (!logfile.empty()) {
	// Needs to be reset in destructor, if not => segfault.
	cout.rdbuf(out.rdbuf());
  }
  else if (silent) {
	cout.rdbuf(0);
  }

  m_cc_tol.set_log(cc_tol);

  m_n = n;
  m_K = K;
  m_cc_limit = cc_limit;

  m_tau_simple = new Treal*[n];
  m_tau_cc = new unordered_map< bm64, Treal >[n];
  m_score_array = new Treal[n * ((bm64) 1 << K)];

  m_C = new int*[n];
  isums = new GroundSetIntersectSums*[n];

  bm64 i = 0;
  int j = 0;

  cout << "Number of candidate parent sets after pruning (unpruned 2^K = " << (1L << K) << "):" << endl << endl;;
  cout << "node\tpsets\tratio" << endl;

  for (int v = 0; v < n; ++v) {
    m_tau_simple[v] = new Treal[ (bm32) 1 << K];
	m_tau_cc[v] = unordered_map< bm64, Treal >();
    m_C[v] = new int[K];
    isums[v] = new GroundSetIntersectSums(K, &score_array[v * ((bm64) 1 << K)], isum_tol);

	cout << v << "\t" << isums[v]->s.size() << "\t" << (double) isums[v]->s.size() / (1L << K) << endl;

    for (bm32 s = 0; s < (bm32) 1 << K; ++s) {
      m_tau_simple[v][s].set_log(score_array[i++]);
    }

    for (int c = 0; c < K; ++c) {
      m_C[v][c] = C[j++];
    }
  }
  cout << endl;

  for (bm64 i = 0; i < n * ((bm64) 1 << K); ++i) { m_score_array[i].set_log(score_array[i]);}

  precompute_tau_simple();
  precompute_tau_cc_basecases();
  precompute_tau_cc();

  int cc = 0;
  for (int v = 0; v < n; ++v) {
    cc += m_tau_cc[v].size();
  }
  cout << "Number of score sums stored in cc cache: " << cc << endl << endl;
}

CandidateRestrictedScore::~CandidateRestrictedScore() {
  delete [] m_tau_cc;
  for (int v = m_n - 1; v > -1; --v) {
    delete[] m_tau_simple[v];
    delete[] m_C[v];
    delete isums[v];
  }
  delete [] isums;
  delete [] m_score_array;
  cout.rdbuf(coutbuf);
}

void CandidateRestrictedScore::reset_cout() {
  cout.rdbuf(coutbuf);
}

void CandidateRestrictedScore::precompute_tau_simple() {

  for (int v = 0; v < m_n; ++v) {
    fzt_inpl(m_tau_simple[v], m_K);
  }
}

void CandidateRestrictedScore::precompute_tau_cc_basecases() {

  int count = 0;
  for (int v = 0; v < m_n; ++v) {
    for (int k = 0; k < m_K; ++k) {
      bm32 j = 1 << k;
      bm32 U_minus_j = (( (bm32) 1 << m_K) - 1) & ~j;
      Treal* tmp = new Treal[1 << (m_K - 1)];

	  tmp[0] = m_score_array[v*( (bm64) 1 << m_K) + j]; // Make indexing inline function?

      // Iterate over subsets of U_minus_j.
      for (bm32 S = U_minus_j; S > 0; S = (S-1) & U_minus_j) {
		tmp[dkbit_32(S, k)] = m_score_array[v*( (bm64) 1 << m_K) + (S | j)];
      }
      fzt_inpl(tmp, (m_K - 1));

      for (bm32 S = 0; S < (bm32) 1 << (m_K - 1); ++S) {
		bm32 U = ikbit_32(S, k, 1);
		if (m_tau_simple[v][U] < m_cc_tol * m_tau_simple[v][U & ~j]) {
		  m_tau_cc[v].insert({(bm64) U << 32 | j, tmp[S]});

		  count++;
		  if (count >= m_cc_limit) {
			delete[] tmp;
			return;
		  }
		}
      }

      delete[] tmp;
    }
  }
}


void CandidateRestrictedScore::precompute_tau_cc() {

  int count[1] = {0};
  for (int v = 0; v < m_n; ++v) {
    *count += m_tau_cc[v].size();
  }

  // With Insurance data, this order seems about 25% faster than having v loop after count check
  for (int v = 0; v < m_n; ++v) {
    for (int T_size_limit = 1; T_size_limit <= m_K; ++T_size_limit) {
      for (bm32 U = 1; U < (bm32) 1 << m_K; ++U) {
		if (count_32(U) < T_size_limit) {
		  continue;
		}
		rec_it_dfs(v, U, U, 0, 0, T_size_limit, count);
		if (*count >= m_cc_limit) {
		  return;
		}
      }
    }
  }
}


Treal CandidateRestrictedScore::sum(int v, bm32 U) {
  return isums[v]->scan_sum(U);
  return isums[v]->scan_sum(U);
}


Treal CandidateRestrictedScore::sum(int v, bm32 U, bm32 T, bool isum) {

  if (isum) {
	return isums[v]->scan_sum(U, T);
  }

  if (U == 0 && T == 0) {
    return m_score_array[v*((bm64) 1 << m_K)];
  }

  if (T == 0) {
	Treal s; s = 0.0;
	return s;
  }

  if (m_tau_cc[v].count((bm64) U << 32 | T)) {
    return m_tau_cc[v][(bm64) U << 32 | T];
  }

  if (m_tau_simple[v][U] < m_cc_tol * m_tau_simple[v][U & ~T]) {
    return isums[v]->scan_sum(U, T);
  }

  return m_tau_simple[v][U] - m_tau_simple[v][U & ~T];
}

pair<bm32, double> CandidateRestrictedScore::sample_pset(int v, bm32 U, bm32 T, double wcum) {
  return isums[v]->scan_rnd(U, T, wcum);
}

pair<bm32, double> CandidateRestrictedScore::sample_pset(int v, bm32 U, double wcum) {
  return isums[v]->scan_rnd(U, wcum);
}


void CandidateRestrictedScore::rec_it_dfs(int v, bm32 U, bm32 R, bm32 T, int T_size, int T_size_limit, int *count) {

  if (*count < m_cc_limit) {
    if (T > 0) {
      if (m_tau_simple[v][U] < m_cc_tol * m_tau_simple[v][U & ~T]) {
		bm32 T1 = T & ~T;
		bm32 U1 = U & ~T1;
		bm32 T2 = T & ~T1;
		if (!m_tau_cc[v].count((bm64) U << 32 | T)) {
		  ++*count;

		  Treal sum1, sum2;
		  sum1 = sum(v, U, T1);
		  sum2 = sum(v, U1, T2);
		  m_tau_cc[v][(bm64) U << 32 | T] = sum1 + sum2;
		}
      }
      else {
		return;
      }
    }
    while (R && T_size < T_size_limit) {
      bm32 B = lsb_32(R);
      R = R ^ B;
      T_size = count_32(T | B);
      rec_it_dfs(v, U, R, T | B, T_size, T_size_limit, count);
    }
  }
}
