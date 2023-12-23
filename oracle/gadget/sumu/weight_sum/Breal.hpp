// Compile with g++ -Wall -O3 -ffast-math
// If used within a tight loop, also try out -funroll-loops
#ifndef BREAL_HPP
#define BREAL_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>

// Class B2real
class B2real { 
    public:
	uint64_t a; int64_t b; 
	
	void		set_log (double z);
	void		set_logl(long double z);
	void		set(int64_t z);
	void		set(unsigned z){ set((int64_t)z); };
	void		set(double z);
	void		set(long double z);
	double		get_log();
	double		get_double();
	long double	get_ldouble();
	int64_t		get_lint();

	B2real& operator=(const int64_t z){ set(z); return *this; }
	B2real& operator=(const double z){ set(z); return *this; }
	B2real& operator=(const long double z){ set(z); return *this; }
	
	inline B2real& operator+=(const B2real y){
		int64_t d = b - y.b;
		if ( d > 63 || y.a == 0){ return *this; }
		if (-d > 63 ||   a == 0){ *this = y; return *this; }
		if ( d >= 0){ a += (y.a >> d); } else { a = y.a + (a >> -d); b = y.b; }
		while (a & 0x8000000000000000){ a >>= 1; b += 1; } return *this;
	}
	inline B2real& operator|=(const B2real y){
		int64_t d = b - y.b; if (d > 63) return *this;
		if (d >= 0){ a += (y.a >> d); } 
		else if (-d > 63){ *this = y; return *this; } else { a = y.a + (a >> -d); b = y.b; }
		while (a & 0x8000000000000000){ a >>= 8; b += 8; } return *this;
	}	
	inline B2real& operator<<=(int n){ b += n; return *this; }
	inline B2real& operator>>=(int n){ b -= n; return *this; }
};

inline void B2real::set_log (double z)     { // Assumes z is the natural log of the number to be represented.
	b = (int64_t)(z * log2 (exp (1.0))) - 62; a = (int64_t) exp (z - ((double)b) * log (2.0));
	while   (a & 0x8000000000000000) { a >>= 1; b++; }
	while (!(a & 0x4000000000000000)){ a <<= 1; b--; }
}
inline void B2real::set_logl(long double z){ // Assumes z is the natural log of the number to be represented.
	b = (int64_t)(z * log2l(expl(1.0))) - 62; a = (int64_t) expl(z - b * logl(2.0));
	while   (a & 0x8000000000000000) { a >>= 1; b++; }
	while (!(a & 0x4000000000000000)){ a <<= 1; b--; }
}
inline void B2real::set(int64_t z){
	if (z == 0){ a = 0; b = -(1LL << 62); return; }	
	b = 0;
	while (  z & 0x8000000000000000) { z >>= 1; b++; } // Truncates the low bits of z if needed.
	while (!(z & 0x4000000000000000)){ z <<= 1; b--; }
	a = (uint64_t)(z);
}
inline void   B2real::set(double z)     { if (z <= (double) 0.0){ set((int64_t) 0);} else set_log (log (z)); }
inline void   B2real::set(long double z){ if (z <= (long double) 0.0){ set((int64_t) 0);} else set_logl(logl(z)); }
inline double B2real::get_log()         { return (double)(log(a) + b*log(2.0)); }
inline double B2real::get_double()      { return (double)(a * pow(2, b)); }
inline long double B2real::get_ldouble(){ return (long double)((long double)(a) * powl(2, (long double)(b))); }
inline int64_t B2real::get_lint()       { int64_t aL = a; if (b >= 0) return aL << b; return aL >> -b; }


inline B2real operator+(B2real x, B2real y){ // Could speed up this by 20 % using certain assumptions.
	int64_t d = x.b - y.b; uint64_t z; int p;
	if (d >= 0){ if ( d > 63){ return x; } z = x.a + (y.a >>  d); p = x.b; } 
	else       { if (-d > 63){ return y; } z = y.a + (x.a >> -d); p = y.b; }
	if (z & 0x8000000000000000){ z >>= 1; p += 1; } return { z, p };	
}
inline B2real operator-(B2real x, B2real y){
	int64_t d = x.b - y.b; uint64_t z; int p;
	if (d >= 0){ if ( d > 63){ return x; } z = x.a - (y.a >>  d); p = x.b; } 
	else       { if (-d > 63){ return y; } z = (x.a >> -d) - y.a; p = y.b; }
	if (z & 0x4000000000000000){ z <<= 1; p -= 1; } return { z, p };	
}
inline uint64_t operator^(B2real x, B2real y){ // Usage: "if (x ^ y) { then x is not close to y  }".
	int64_t d = x.b - y.b;  uint64_t z; 
	if (d >= 0){ if ( d > 63){ return x.a; } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 63){ return y.a; } z = y.a ^ (x.a >> -d); }
	return z & 0xFFFFFF0000000000; // Return the difference in the matched most significant 24 bits.	
}
inline uint64_t diff(uint64_t m, B2real x, B2real y){
	int d = x.b - y.b;  uint32_t z; 
	if (d >= 0){ if ( d > 63){ return x.a; } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 63){ return y.a; } z = y.a ^ (x.a >> -d); }
	return z & (((1LL << m) - 1LL) << (63 - m)); // Return the difference in the matched most significant m bits.	
}
inline bool operator==(B2real x, B2real y){ // Usage: "if (x == y) { then x is equal to y  }".
	int64_t d = x.b - y.b;  uint64_t z; 
	if (d >= 0){ if ( d > 63){ return (x.a == 0 && y.a == 0); } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 63){ return (y.a == 0 && x.a == 0); } z = y.a ^ (x.a >> -d); }
	return (z == 0); // Return true if no difference after matching the exponents.	
}
inline B2real operator*(B2real x, B2real y){ 
	uint64_t x0 = x.a & 0x7fffffff, y0 = y.a & 0x7fffffff; x.a >>= 31; y.a >>= 31; // Ignore the 31 lsb, 32 msb left.
	x0 *= y.a; y0 *= x.a; uint64_t z = x.a * y.a + (x0 >> 31) + (y0 >> 31); int64_t p = x.b + y.b + 62;
	while (z & 0x8000000000000000){ z >>= 1; ++p; } return { z, p };	
}

inline bool operator< (const B2real x, const B2real y){ return x.b < y.b || (x.b == y.b && x.a < y.a); }
inline bool operator> (const B2real x, const B2real y){ return   y < x;  }
inline bool operator<=(const B2real x, const B2real y){ return !(y < x); }
inline bool operator>=(const B2real x, const B2real y){ return !(x < y); }

inline B2real operator+(B2real x, int64_t w){ B2real y; y = w; return x + y; }
inline B2real operator+(int64_t w, B2real y){ B2real x; x = w; return x + y; }
inline B2real operator-(B2real x, int64_t w){ B2real y; y = w; return x - y; }
inline B2real operator-(int64_t w, B2real y){ B2real x; x = w; return x - y; }
inline B2real operator*(B2real x, int64_t w){ B2real y; y = w; return x * y; }
inline B2real operator*(int64_t w, B2real y){ B2real x; x = w; return x * y; }

inline bool operator< (const B2real x, const int64_t w){ B2real y; y = w; return x <  y; }
inline bool operator< (const int64_t w, const B2real y){ B2real x; x = w; return x <  y; }
inline bool operator> (const B2real x, const int64_t w){ B2real y; y = w; return x >  y; }
inline bool operator> (const int64_t w, const B2real y){ B2real x; x = w; return x >  y; }
inline bool operator<=(const B2real x, const int64_t w){ B2real y; y = w; return x <= y; }
inline bool operator<=(const int64_t w, const B2real y){ B2real x; x = w; return x <= y; }
inline bool operator>=(const B2real x, const int64_t w){ B2real y; y = w; return x >= y; }
inline bool operator>=(const int64_t w, const B2real y){ B2real x; x = w; return x >= y; }
 
inline B2real operator+(B2real x, double w){ B2real y; y = w; return x + y; }
inline B2real operator+(double w, B2real y){ B2real x; x = w; return x + y; }
inline B2real operator-(B2real x, double w){ B2real y; y = w; return x - y; }
inline B2real operator-(double w, B2real y){ B2real x; x = w; return x - y; }
inline B2real operator*(B2real x, double w){ B2real y; y = w; return x * y; }
inline B2real operator*(double w, B2real y){ B2real x; x = w; return x * y; }

inline bool operator< (const B2real x, const double w){ B2real y; y = w; return x <  y; }
inline bool operator< (const double w, const B2real y){ B2real x; x = w; return x <  y; }
inline bool operator> (const B2real x, const double w){ B2real y; y = w; return x >  y; }
inline bool operator> (const double w, const B2real y){ B2real x; x = w; return x >  y; }
inline bool operator<=(const B2real x, const double w){ B2real y; y = w; return x <= y; }
inline bool operator<=(const double w, const B2real y){ B2real x; x = w; return x <= y; }
inline bool operator>=(const B2real x, const double w){ B2real y; y = w; return x >= y; }
inline bool operator>=(const double w, const B2real y){ B2real x; x = w; return x >= y; }

#endif
