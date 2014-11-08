#include <Tools/StringConversion.hpp>

#include <cstdlib> // atoi

int StringConversion::atoi(const std::string& str)
{
    return std::atoi(str.c_str());
}

float StringConversion::atof(const std::string& str)
{
    return std::atof(str.c_str());
}

double StringConversion::atod(const std::string& str)
{
    return std::strtod(str.c_str(), nullptr);
}
