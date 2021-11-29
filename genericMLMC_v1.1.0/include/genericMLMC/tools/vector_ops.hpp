#ifndef GENERICMLMC__TOOLS__VECTOR_OPS_HPP
#define GENERICMLMC__TOOLS__VECTOR_OPS_HPP
#pragma once

#include <vector>
#include <cassert>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <functional>

template<class T, class S>
std::vector<T>& operator*=(std::vector<T>& x, S alpha) {
    std::transform(x.begin(), x.end(), x.begin(), [alpha](T xx){return T(alpha)*xx;});
    return x;
}

template<class T>
std::vector<T> operator*(T alpha, const std::vector<T>& x) {
    std::vector<T> ret(x);
    std::transform(ret.begin(), ret.end(), ret.begin(), [alpha](T xx){return alpha*xx;});
    return ret;
}

template<class T>
std::vector<T> operator*(const std::vector<T>& x, T alpha) {
    return operator*(alpha, x);
}

template<class T>
std::vector<T> operator+(const std::vector<T>& x, const std::vector<T>& y) {
    assert(x.size() == y.size());
    size_t n = x.size();
    std::vector<T> ret(n);
    std::transform(x.begin(), x.end(), y.begin(), ret.begin(), std::plus<T>());
    return ret;
}

template<class T>
std::vector<T>& operator+=(std::vector<T>& x, const std::vector<T>& y) {
    assert(x.size() == y.size());
    size_t n = x.size();
    for (size_t i=0; i<n; ++i)
        x[i] += y[i];
    return x;
}

namespace detail {


/// read and write vectors to file (binary mode)
/// format: first entry is length of vector; then comes the content

/** writes vector to file */
template<class T>
bool writeVector(std::vector<T>& vector, std::string filename){
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
    if (file.is_open()) {
        unsigned int N = vector.size();
        file.write(reinterpret_cast<char*>(&N),sizeof(unsigned int)); // number of elements
        file.write(reinterpret_cast<char*>(&vector.front()),vector.size()*sizeof(T)); // leave out first element
        file.close();
        return true;
    } else {
        throw std::runtime_error(std::string("could not open file: ")+filename);
        return false;
    }
}

/** reads file to vector. if n negative, reads N from file, else does not. */
template<class T>
bool readVector(std::vector<T>& vector, std::string filename, int N=-1){
    // TODO: check if file exists. if not, throw an exception
    std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
    if (file.is_open()) {
        if (N<0)
            file.read(reinterpret_cast<char*>(&N),sizeof(unsigned int)); // get number of elements
        vector.resize(N);
        file.read(reinterpret_cast<char*>(&vector.front()),N*sizeof(T)); // leave out first element
        file.close();
        return true;
    } else {
        throw std::runtime_error(std::string("could not open file: ")+filename);
        return false;
    }
}


} // namespace detail


#endif /* end of include guard: GENERICMLMC__TOOLS__VECTOR_OPS_HPP */
