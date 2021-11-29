#ifndef GENERICMLMC__TOOLS__CAN_CALL_HPP
#define GENERICMLMC__TOOLS__CAN_CALL_HPP
#pragma once

#include <type_traits>

#include <boost/mpi/communicator.hpp>

namespace MLMC {
namespace tools {

/** @file
 * @brief This file implements the traits can_call and return_type.
 *
 * from:
 *http://stackoverflow.com/questions/22882170/c-compile-time-predicate-to-test-if-a-callable-object-of-type-f-can-be-called
 * see also: https://github.com/sth/callable.hpp
 */

struct can_call_test {
    template <typename F, typename... A>
    static decltype(std::declval<F>()(std::declval<A>()...), std::true_type()) f(int);

    template <typename F, typename... A>
    static std::false_type f(...);
};

template <typename F, typename... A>
struct can_call : decltype(can_call_test::f<F, A...>(0)) {};

template <typename F, typename... A>
struct can_call<F(A...)> : can_call<F, A...> {};

template <typename... A, typename F>
constexpr can_call<F, A...> is_callable_with(F&&) {
    return can_call<F(A...)>{};
}

// for MLMC: find out correct return type

template <typename F, typename T, bool twoargs, bool threeargs>
struct return_type_helper;

template <typename F, typename T>
struct return_type_helper<F, T, true, false> {
    using type = decltype(std::declval<F>()(std::declval<T>(), int()));
};

template <typename F, typename T>
struct return_type_helper<F, T, false, true> {
    using type = decltype(std::declval<F>()(std::declval<T>(), int(), boost::mpi::communicator()));
};

template <typename F, typename T>
struct return_type_helper<F, T, true, true> {
    using type = decltype(std::declval<F>()(std::declval<T>(), int(), boost::mpi::communicator()));
};

template <typename F, typename T>
struct return_type {
    using type = typename return_type_helper<F, T,
                                            (bool)can_call<F(T, int)>(),
                                            (bool)can_call<F(T, int, boost::mpi::communicator)>()>::type;
};

} // namespace tools
} // namespace MLMC

#endif /* end of include guard: GENERICMLMC__TOOLS__CAN_CALL_HPP */
