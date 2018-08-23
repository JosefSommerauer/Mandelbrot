//       $Id: snippet-4.h 29419 2015-10-13 06:46:29Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2015-WS/Ablauf/src/Fractals/src/snippet-4.h $
// $Revision: 29419 $
//     $Date: 2015-10-13 08:46:29 +0200 (Di., 13 Okt 2015) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
//   Remarks: Get an RGB-value from a light beam's wavelength.
//     Types:
// Functions: pfc::rgb_from_wavelength, pfc::rgb_from_wavelength_pm

#ifndef RGB_FROM_WAVELENGTH
#define RGB_FROM_WAVELENGTH

#include <algorithm>

namespace pfc {

// macro to cast int to color type
template <class T, class M> M get_member_type(M T:: *);
#define GET_TYPE_OF(mem) decltype(get_member_type(mem))

template <typename rgb_t> inline void rgb_from_wavelength_pm (rgb_t & col, int wavelength, bool const gray = false) {
   typedef decltype (col.red)   red_t;     // rgb_t::red
   typedef decltype (col.green) green_t;   // rgb_t::green
   typedef decltype (col.blue)  blue_t;    // rgb_t::blue

   static_assert (
      std::is_same <red_t, green_t>::value && std::is_same <red_t, blue_t>::value,
      "The color components of 'rgb_from_wavelength::rgb_t' must be of identical types."
   );

   static_assert (
      std::is_integral <red_t>::value && std::is_unsigned <red_t>::value,
      "The color components of 'rgb_from_wavelength::rgb_t' must be unsigned integral values."
   );

   wavelength = std::min (std::max (wavelength, 380000), 780000);

   unsigned r = 0;
   unsigned g = 0;
   unsigned b = 0;

   if (wavelength <  440000) { r = (440000 - wavelength) / 60; b = 1000; } else
   if (wavelength <  490000) { g = (wavelength - 440000) / 50; b = 1000; } else
   if (wavelength <  510000) { g = 1000; b = (510000 - wavelength) / 20; } else
   if (wavelength <  580000) { r = (wavelength - 510000) / 70; g = 1000; } else
   if (wavelength <  645000) { r = 1000; g = (645000 - wavelength) / 65; } else
   if (wavelength <= 780000) { r = 1000;                                 }

   double f = 0;

   if (wavelength <  420000) { f = -0.006350 + 0.00000001750 * wavelength; } else
   if (wavelength <  701000) { f =  0.001000;                              } else
   if (wavelength <= 780000) { f =  0.007125 - 0.00000000875 * wavelength; }

   double const alpha = 0.8;

   col.red   = red_t   (std::numeric_limits <red_t>  ::max () * std::min (std::pow (r * f, alpha), 1.0));
   col.green = green_t (std::numeric_limits <green_t>::max () * std::min (std::pow (g * f, alpha), 1.0));
   col.blue  = blue_t  (std::numeric_limits <blue_t> ::max () * std::min (std::pow (b * f, alpha), 1.0));

   if (gray) {
      col.red   =
      col.green =
      col.blue  = static_cast <red_t> ((col.red + col.green + col.blue) / 3);
   }
}

template <typename rgb_t> inline void rgb_from_wavelength (rgb_t & col, int const wavelength, bool const gray = false) {
   rgb_from_wavelength_pm <rgb_t> (col, wavelength * 1000, gray);
}

template <typename rgb_t> inline void rgb_from_wavelength (rgb_t & col, double const x, bool const gray = false) {
   rgb_from_wavelength_pm <rgb_t> (col, static_cast <int> (380000 + (780000 - 380000) * x), gray);
}

template <typename rgb_t> inline rgb_t get_rgb_wavelength(int const n, size_t iter_max) {
	rgb_t col;

	int x = ((((double)n)*(780 - 380)) / iter_max) + 380;

	rgb_from_wavelength_pm<rgb_t>(col, x * 1000, false);
	return col;
}

// extra color generaters

template <typename rgb_t> inline rgb_t get_rgb_smooth(int const n, size_t iter_max) {
	// map n on the 0..1 interval
	double t = (double)n / (double)iter_max;

	// Use smooth polynomials for r, g, b
	int b = (int)(9 *  (1 - t)*t      *t      *t * 255);
	int g = (int)(15 * (1 - t)*(1 - t)*t      *t * 255);
	int r = (int)(8.5* (1 - t)*(1 - t)*(1 - t)*t * 255);
	return{ 
		(GET_TYPE_OF(&rgb_t::red)) r,
		(GET_TYPE_OF(&rgb_t::blue)) b, 
		(GET_TYPE_OF(&rgb_t::green)) g 
	};
}

template <typename rgb_t> inline rgb_t get_rgb_piecewise_linear(int const n, int iter_max) {
 	int N = 256; // colors per element
 	int N3 = N * N * N;
 	// map n on the 0..1 interval (real numbers)
 	double t = (double)n / (double)iter_max;
 	// expand n on the 0 .. 256^3 interval (integers)
    int n2 = (int)(t * (double)N3);

 	int b = n2 / (N * N);
 	int nn = n2 - b * N * N;
 	int r = nn / N;
 	int g = nn - r * N;
	return{ 
		(GET_TYPE_OF(&rgb_t::red)) r,
		(GET_TYPE_OF(&rgb_t::blue)) b, 
		(GET_TYPE_OF(&rgb_t::green)) g 
	};
}

template <typename rgb_t> inline rgb_t get_black_white(int const n, size_t iter_max) {

	int b = (n % 2) ? 0xFF : 0x00;

	return{ 
		(GET_TYPE_OF(&rgb_t::red)) b,
		(GET_TYPE_OF(&rgb_t::blue)) b, 
		(GET_TYPE_OF(&rgb_t::green)) b 
	};
}

}   // namespace pfc

#endif
