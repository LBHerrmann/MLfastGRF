#ifndef GENERICMLMC__TOOLS__PARALLEL_HPP
#define GENERICMLMC__TOOLS__PARALLEL_HPP
#pragma once

#ifndef MLMC_SERIAL_FLAG
#include <boost/mpi.hpp>
#endif

namespace MLMC {
namespace tools {

#ifndef MLMC_SERIAL_FLAG
/**
 * Since newer versions of boost::mpi add a specialization of reduce for std::vectors,
 * one can no longer give an operation on the type being transmitted.
 * In order to still support the VectorSpaceElement concept,
 * we need a hack that uses plus<T> where T is the inner type of the vector.
 */

template<typename T, typename Op>
void
reduce_helper(const boost::mpi::communicator& comm, const T& in_value, T& out_value, Op op, int root)
{
  if (comm.rank() == root)
    boost::mpi::detail::reduce_impl(comm, &in_value, 1, &out_value, op, root,
                        boost::mpi::is_mpi_op<Op, T>(), boost::mpi::is_mpi_datatype<T>());
  else
    boost::mpi::detail::reduce_impl(comm, &in_value, 1, op, root,
                        boost::mpi::is_mpi_op<Op, T>(), boost::mpi::is_mpi_datatype<T>());
}

// specialize on vector and directly pass std::plus<T>
template<typename T, typename Op> 
void 
reduce_helper(const boost::mpi::communicator & comm, const std::vector<T> & in_values, std::vector<T> & out_values, Op op, int root) 
{
  out_values.resize(in_values.size());
  reduce(comm, &in_values.front(), in_values.size(), &out_values.front(), std::plus<T>(),
         root);
}
#endif

/**
 * @returns the number of processors required to run
 * @param Ml number of samples per level
 * @param Pl number of sample solvers per level
 * @param Dl number of processesors per sample solver
 */
template <class T1, class T2>
T2 num_required_processors(std::vector<T1>& Ml, std::vector<T2>& Pl, std::vector<T2>& Dl) {
    T2 ret = 0;
    for (int l = 0; l < Ml.size(); l++) ret += std::min<long>(Pl[l], Ml[l]) * Dl[l];
    return ret;
}

/**
 * Checks if arguments Ml, Pl, Dl contain sensible values. Throws if not the case.
 */
template <class T1, class T2>
void check_arguments(std::vector<T1>& Ml, std::vector<T2>& Pl, std::vector<T2>& Dl) {
    std::stringstream err;
    bool have_err = false;
    if (Pl.size() != Ml.size()) {
        err << "Incorrect size of vectors: Pl and Ml should be of same size" << std::endl;
        have_err = true;
    }
    if (Pl.size() != Dl.size()) {
        err << "Incorrect size of vectors: Pl and Dl should be of same size" << std::endl;
        have_err = true;
    }
    for (int i = 0; i < Pl.size(); ++i)
        if (Pl[i] < 1) {
            err << "The value of Pl must be between positive (i=" << i << ")." << std::endl;
            have_err = true;
        }
    if (have_err) throw std::runtime_error(err.str());
}

/**
 * @brief   computes the level of the MPI process with given rank.
 *
 * @param Pl Number of samplesolvers per level (l=0,...,L)
 * @param Dl Number of processes per samplesolver
 * @param rank rank of the process to get level of
 * @returns the level of the current process. Is L+1 if rank not used in simulation.
 */
template <class T>
int get_level(std::vector<T>& Pl, std::vector<T>& Dl, int rank) {
    int sum = 0;
    for (int l = 0; l < Pl.size(); l++) {
        sum += Pl[l] * Dl[l];
        if (rank < sum) return l;
    }
    return Pl.size() + 1;
}

#ifndef MLMC_SERIAL_FLAG
/** Overload in case a communicator is passed instead of the rank (which compiles). */
template <class T>
bool get_level(std::vector<T>& Pl, std::vector<T>& Dl, boost::mpi::communicator comm) {
    return get_level(Pl, Dl, comm.rank());
}
#endif

/**
 * @brief determines if the given rank is a level root or not.
 *
 * @param Pl Number of samplesolvers per level (l=0,...,L)
 * @param Dl Number of processes per samplesolver
 * @param rank rank of the process to get level of
 * @returns true if the rank is a level root
 */
template <class T>
bool is_level_root(std::vector<T>& Pl, std::vector<T>& Dl, int rank) {
    int sum = 0;
    for (int l = 0; l < Pl.size(); l++) {
        if (rank == sum) return true;
        sum += Pl[l] * Dl[l];
    }
    return false;
}

#ifndef MLMC_SERIAL_FLAG
/** Overload in case a communicator is passed instead of the rank (which compiles). */
template <class T>
bool is_level_root(std::vector<T>& Pl, std::vector<T>& Dl, boost::mpi::communicator comm) {
    return is_level_root(Pl, Dl, comm.rank());
}
#endif

/**
 * @brief computes the number of samples M_l^i for samplesolver i
 *
 * @param Ml number of samples on entire level l
 * @param Pl number of samplesolvers on level l
 * @param solverindex index of the samplesolver on level l
 * @returns the number of samples M_l per sample solver
 */
template <class T>
T Ml_per_solver(T Ml, int Pl, int solverindex) {
    T ret = Ml / Pl;
    if (solverindex < Ml % Pl) ret++;
    return ret;
}

}  // namespace tools
}  // namespace MLMC


#endif /* end of include guard: GENERICMLMC__TOOLS__PARALLEL_HPP */
