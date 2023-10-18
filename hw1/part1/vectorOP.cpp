#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  __pp_vec_float x, result;
  __pp_vec_int y, count;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float maxFloat = _pp_vset_float(9.999999f);
  __pp_mask maskAll, maskIsEqualZero, maskIsNotEqualZero, maskIsPositive, maskIsBiggerThanMaxFloat;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    if (i + VECTOR_WIDTH > N)
      maskAll = _pp_init_ones(N - i);
    else
      maskAll = _pp_init_ones();
    maskIsEqualZero = _pp_init_ones(0);
    maskIsNotEqualZero = _pp_init_ones(0);
    maskIsPositive = _pp_init_ones(0);
    maskIsBiggerThanMaxFloat = _pp_init_ones(0);

    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    _pp_vload_int(y, exponents + i, maskAll); // y = exponents[i];

    _pp_veq_int(maskIsEqualZero, y, zero, maskAll); // if (y == 0)

    _pp_vset_float(result, 1.f, maskIsEqualZero); // output[i] = 1.f

    maskIsNotEqualZero = _pp_mask_not(maskIsEqualZero); // else

    _pp_vmove_float(result, x, maskIsNotEqualZero); // result = x;

    _pp_vsub_int(count, y, one, maskIsNotEqualZero); // count = y - 1;

    // if count > 0
    _pp_vgt_int(maskIsPositive, count, zero, maskIsNotEqualZero); 
    
    while (_pp_cntbits(maskIsPositive)) {
      _pp_vmult_float(result, result, x, maskIsPositive);	// result *= x;
	    _pp_vsub_int(count, count, one, maskIsPositive);	// count--;
	    _pp_vgt_int(maskIsPositive, count, zero, maskIsPositive);
    }

    _pp_vgt_float(maskIsBiggerThanMaxFloat, result, maxFloat, maskAll); // if (result > 9.999999f)

    _pp_vset_float(result, 9.999999f, maskIsBiggerThanMaxFloat);   // output[i] = 9.999999f;

    _pp_vstore_float(output + i, result, maskAll);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}