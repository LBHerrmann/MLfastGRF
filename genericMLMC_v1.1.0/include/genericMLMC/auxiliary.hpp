#ifndef GENERICMLMC__AUXILIARY_HPP
#define GENERICMLMC__AUXILIARY_HPP

#pragma once

#include <cassert>
#include <vector>
#include <functional>

namespace MLMC {

/*
 * @brief  Projects an MLMC sample 'D' from the  level 'level' to level 'level-1'
 * @returns MLMC sample 'D' projected from MLMC level 'level' to MLMC level 'level-1'
 */
template <class T>
T project_sample(T&& D, int level) {
    return D;
}

/*
 * @brief  Projects the MLMC QoI 'D' from the  level 'level-1' to level 'level'
 * @returns MLMC QoI 'D' projected from MLMC level 'level-1' to MLMC level 'level'
 */
template <class T>
T project_solution(T& D, int level) {
    return D;
}

/**
 * Combine function.
 * Requires only vector space operations on the type T
 * (i.e. x+y and a*y, where x,y are of type T and a is a double)
 */
template <class T>
void combine(T& Sum, const T& X, const T& Y, const int l) {
    auto temp = project_solution(Y, l);
    temp = X + (-1.) * temp;
    Sum = Sum + temp;
}

}  // end of namespace MLMC


#endif /* end of include guard: GENERICMLMC__AUXILIARY_HPP */
