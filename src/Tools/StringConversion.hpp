#pragma once

#include <sstream>

/**
 * Fallbacks for standards functions 
 * (missing in MinGW for example) 
**/
namespace StringConversion
{

/**
 * Fallback for std::to_string
 * @param val Value to convert to string
 * @return Conversion to std::string
**/
template<typename T>
std::string to_string(T val)
{
    std::stringstream out;
    out << val;
    return out.str();
}

/**
 * Fallback for std::atoi
 * @param str String to convert
 * @return Conversion to int
**/
int atoi(const std::string& str);

/**
 * Fallback for std::atof
 * @param str String to convert
 * @return Conversion to float
**/
float atof(const std::string& str);

/**
 * Fallback for std::atod
 * @param str String to convert
 * @return Conversion to double
**/
double atod(const std::string& str);

};
