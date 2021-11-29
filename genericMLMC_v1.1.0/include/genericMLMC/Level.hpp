#ifndef GENERICMLMC__LEVEL_HPP
#define GENERICMLMC__LEVEL_HPP
#pragma once

#include <limits>
#include <type_traits>
#include <cassert>
#include <functional>
// boost includes
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
// genericMLMC includes
#include <genericMLMC/auxiliary.hpp>
#include <genericMLMC/tools/can_call.hpp>
#include <genericMLMC/tools/parallel.hpp>

namespace MLMC {

/**
 * @brief- Provides the compute() function for serial integrands. Implementation uses
 * CRTP - Curiously Recurring Template Pattern
 */
template <class Derived, class result_type, bool is_parallel = false>
struct compute_helper {
    /**
     *  @brief - compute MLMC given serial integrand
     */
    result_type compute() {
        result_type sum, groupsum;

        long Ml = tools::Ml_per_solver(static_cast<Derived*>(this)->Ml_, static_cast<Derived*>(this)->Pl_,
                                      static_cast<Derived*>(this)->get_solver_index());
        if (Ml == 0) {
            // TODO: shouldn't reach this point; in general, would need way to initialize groupsum correctly.
            return groupsum;
        }

        // level without trailing _ is temporary variable local to this function
        int level = static_cast<Derived*>(this)->get_level();

        if (level == 0) {
            // do first sample before loop to correctly initialize 'sum'
            auto sample = static_cast<Derived*>(this)->sample_generator_(level);
            sum = static_cast<Derived*>(this)->integrand_(sample, level);
            // loop
            for (long samplecounter = 1; samplecounter < Ml; ++samplecounter) {
                auto sample = static_cast<Derived*>(this)->sample_generator_(level);
                auto X_l = static_cast<Derived*>(this)->integrand_(sample, level);

                sum = sum + X_l;
            }
        } else {
            // do first sample before loop to correctly initialize 'sum'
            {  // scope to get rid of sample, X_l, X_l_1 when assignment to sum is done.
                auto sample = static_cast<Derived*>(this)->sample_generator_(level);
                auto X_l = static_cast<Derived*>(this)->integrand_(sample, level);
                auto X_l_1 = static_cast<Derived*>(this)->integrand_(project_sample(std::move(sample), level), level - 1);
                sum = X_l + (-1.) * X_l_1;  // don't need to intialize sum before
            }
            // loop
            for (long samplecounter = 1; samplecounter < Ml; ++samplecounter) {
                auto sample = static_cast<Derived*>(this)->sample_generator_(level);
                auto X_l = static_cast<Derived*>(this)->integrand_(sample, level);
                auto X_l_1 = static_cast<Derived*>(this)->integrand_(project_sample(std::move(sample), level), level - 1);

                combine(sum, X_l, X_l_1, level);
            }
        }

        // reduce to rank 0
        tools::reduce_helper(static_cast<Derived*>(this)->level_comm_, sum, groupsum, std::plus<result_type>(), 0);

        if (static_cast<Derived*>(this)->level_comm_.rank() == 0)
            groupsum = (1. / static_cast<Derived*>(this)->Ml_) * groupsum;

        return groupsum;
    }
};

/**
 * @brief - Provides the compute() function for parallel integrands. Implementation uses
 * CRTP - Curiously Recurring Template Pattern
 */
template <class Derived, class result_type>
struct compute_helper<Derived, result_type, true> {
    /**
     *  @brief - compute MLMC given parallel integrand
     */
    result_type compute() {
        result_type sum, groupsum;

        long Ml = tools::Ml_per_solver(static_cast<Derived*>(this)->Ml_, static_cast<Derived*>(this)->Pl_,
                                      static_cast<Derived*>(this)->get_solver_index());

        boost::mpi::communicator integrand_communicator =
            static_cast<Derived*>(this)->level_comm_.split(static_cast<Derived*>(this)->get_solver_index());
        bool flag = (static_cast<Derived*>(this)->is_root()) && (Ml > 0);
        boost::mpi::communicator reduce_communicator = static_cast<Derived*>(this)->level_comm_.split(flag);

        if (Ml == 0) return groupsum;  // these processes are excluded from the reduce, see above

        // level without trailing _ is temporary variable local to this function
        int level = static_cast<Derived*>(this)->get_level();

        if (level == 0) {
            // do first sample before loop to correctly initialize 'sum'
            auto sample = static_cast<Derived*>(this)->sample_generator_(level);
            sum = static_cast<Derived*>(this)->integrand_(sample, level, integrand_communicator);
            // loop
            for (long samplecounter = 1; samplecounter < Ml; ++samplecounter) {
                // if (integrand_communicator.rank() == 0) { // WRONG!! integrand must be executed on all processors!
                auto sample = static_cast<Derived*>(this)->sample_generator_(level);
                auto X_l = static_cast<Derived*>(this)->integrand_(sample, level, integrand_communicator);

                sum = sum + X_l;
            }
        } else {
            // do first sample before loop to correctly initialize 'sum'
            {  // scope to get rid of sample, X_l, X_l_1 when assignment to sum is done.
                auto sample = static_cast<Derived*>(this)->sample_generator_(level);
                auto X_l = static_cast<Derived*>(this)->integrand_(sample, level, integrand_communicator);
                auto X_l_1 = static_cast<Derived*>(this)
                                 ->integrand_(project_sample(std::move(sample), level), level - 1, integrand_communicator);
                // don't want to separately intialize sum before.
                sum = X_l + (-1.) * X_l_1;
            }
            // loop
            for (long samplecounter = 1; samplecounter < Ml; ++samplecounter) {
                auto sample = static_cast<Derived*>(this)->sample_generator_(level);
                auto X_l = static_cast<Derived*>(this)->integrand_(sample, level, integrand_communicator);
                auto X_l_1 = static_cast<Derived*>(this)
                                 ->integrand_(project_sample(std::move(sample), level), level - 1, integrand_communicator);

                combine(sum, X_l, X_l_1, level);
            }
        }

        if (static_cast<Derived*>(this)->is_root())  // reduce to rank 0
            tools::reduce_helper(reduce_communicator, sum, groupsum, std::plus<result_type>(), 0);

        if (static_cast<Derived*>(this)->level_comm_.rank() == 0)
            groupsum = (1. / static_cast<Derived*>(this)->Ml_) * groupsum;

        return groupsum;
    }
};

/**
 * Represents a level of MLMC, evaluating a function of type FunctionType,
 * mapping sample_type x int -> result_type
 * @tparam FunctionType type of an integrand callable with arguments sample, level, communicator.
 * @tparam SamplerType  type of a sample generator callable with arguments level.
 *
 */
template <class FunctionType, class SamplerType>
class Level
    : public compute_helper<
          Level<FunctionType, SamplerType>,
          typename tools::return_type<FunctionType, decltype(std::declval<SamplerType>()(0))>::type,
          (bool) tools::can_call<FunctionType(decltype(std::declval<SamplerType>()(0)),
          int,
         boost::mpi::communicator)>()
      > {
    // friend to allow use of private variables
    friend class compute_helper<
        Level<FunctionType, SamplerType>,
        typename tools::return_type<FunctionType, decltype(std::declval<SamplerType>()(0))>::type,
        (bool)tools::can_call<FunctionType(decltype(std::declval<SamplerType>()(0)), int, boost::mpi::communicator)>()>;

    /** index of the level the class represents */
    int level_;

    /** number of samples to compute on the whole level (M_l) */
    long Ml_;

    /** number of sample solvers on this level (P_l) */
    int Pl_;

    /** number of cores that each sample_solver uses (D_l) */
    int Dl_;

    /** Index of the samplesolver. Takes values from 0 to Pl[level_]-1. */
    int solver_index_;

    /** identifies if the mpi process is a root */
    bool i_am_a_root_;

    /** MPI communicator for the entire level */
    boost::mpi::communicator level_comm_;

    /**
     * integrand solved with MLMC . Defined here as a callback.
     * Maps a sample_type X int -> result_type
     */
    FunctionType integrand_;

    /** A user defined callback for generating samples. Maps a  int -> sample_type */
    SamplerType sample_generator_;

public:
    // SFINAE used to identify the returntype of the integrand
    typedef decltype(sample_generator_(0)) sample_type;
    typedef typename tools::return_type<FunctionType, sample_type>::type result_type;

    /**
     * Constructor additionally accepting Dl (number of cores per samplesolver)
     * @brief - Level Constructor Given a parallel integrand
     * @param integrand integrand function
     * @param sampler sample generating function
     * @param level Levek allocated in this instance
     * @param Ml Number of samples to be computed in this level
     * @param Pl Number of processes at this level
     * @param Dl Number of cores per sample_solver , default = 1
     * @param groupcomm group communicator (default: world)
     */
    Level(FunctionType integrand, SamplerType sampler, int level, long Ml, int Pl, int Dl = 1,
          boost::mpi::communicator groupcomm = boost::mpi::communicator())
        : integrand_(integrand),
          sample_generator_(sampler),
          Ml_(Ml),
          Pl_(Pl),
          Dl_(Dl),
          level_(level),
          level_comm_(groupcomm),
          solver_index_(std::numeric_limits<int>::max()),
          i_am_a_root_(false) {
        // compute the index of the solver
        int max_rank = Dl_ * Pl_;
        if (level_comm_.rank() >= max_rank) {
            solver_index_ = Pl_;  // invalid value, should be < Pl_
            i_am_a_root_ = false;
            return;
        }
        solver_index_ = level_comm_.rank() / Dl_;
        i_am_a_root_ = level_comm_.rank() % Dl_ == 0;
    }

    /** @returns level index */
    int get_level() { return level_; }

    /** @returns integrand color. If not applicable, is set to -1. */
    int get_solver_index() { return solver_index_; }

    /** @returns true if the rank is a root on the integrand_communicator */
    bool is_root() { return i_am_a_root_; }
};

}  // end of namespace MLMC


#endif /* end of include guard: GENERICMLMC__LEVEL_HPP */
