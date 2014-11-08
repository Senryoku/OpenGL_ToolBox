#pragma once

#include <cmath>

constexpr double pi() { return std::atan(1)*4; }

/**
 * @param val
 * @param min
 * @param max
 * @return 
**/
template<typename ScalarType>
ScalarType clamp(ScalarType val, ScalarType min, ScalarType max)
{
	if(val < min) return min;
	else if(val > max) return max;
	return val;
}

template<typename ScalarType>
ScalarType wrap(ScalarType val, ScalarType min = ScalarType(), ScalarType max = static_cast<ScalarType>(1.f))
{
	while(val < min) val += max - min;
	while(val > max) val -= max - min;
}

template<typename T>
T mod(T x, T y)
{
    static_assert(!std::numeric_limits<T>::is_exact , "mod: floating-point type expected");

    if (0. == y)
        return x;

    double m = x - y * floor(x/y);

    if(y > 0)
    {
        if(m >= y)
            return 0;
			
        if(m < 0)
        {
            if (y + m == y)
                return 0;
            else
				return y + m;
        }
    } else {
        if(m <= y)
            return 0;

        if(m > 0)
        {
            if (y + m == y)
                return 0;
            else
                return y + m;
        }
    }

    return m;
}