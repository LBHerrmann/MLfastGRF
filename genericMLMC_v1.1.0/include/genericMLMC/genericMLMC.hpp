#ifndef GENERICMLMC__GENERICMLMC_HPP
#define GENERICMLMC__GENERICMLMC_HPP

#pragma once

#include <vector>
#include <type_traits>
#include <cassert>
#include <sstream>
#include <chrono>

// SFINAE includes
#include <genericMLMC/tools/can_call.hpp>
#include <genericMLMC/tools/parallel.hpp>

// boost includes
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

// genericMLMC includes
#include <genericMLMC/Level.hpp>
#include <genericMLMC/auxiliary.hpp>

namespace MLMC {

/**
 * The class reads an integrand_ and a sample_generator and implements MLMC for them,
 * @tparam Functiontype
 * @tparam SamplerType
 */

template <class FunctionType, class SamplerType>
class genericMLMC {

    /** callback to the integrand_ to be solved by MLMC */
    FunctionType integrand_;
    /** callback to the sample_generator_ that generates the sample */
    SamplerType sample_generator_;

    /** level of the current process */
    int level_;

    /** Input Communicator provided by the user */
    boost::mpi::communicator world_;

public:
    typedef decltype(sample_generator_(0)) sample_type;
    typedef typename tools::return_type<FunctionType, sample_type>::type result_type;

    /**
    * @brief - Constnuctor for beginning MLMC with default combine function applied. Used if the QoI is scalar or vector
    * @param input_integrand_ - The integrand_ to be solved with MLMC. Passed as a callback.
    * @param input_samplegenerator - Callback for sample generator.
    * @param inputcomm - Input communicator upon which the entire
    */
    genericMLMC(FunctionType input_integrand_, SamplerType input_samplegenerator, boost::mpi::communicator inputcomm)
        : integrand_(input_integrand_),
          sample_generator_(input_samplegenerator),
          world_(inputcomm)
          {}

    /**
    *  @brief  compute the MLMC level given a parallel integrand_.
    *  @returns
    *
    *  1. Initialize the sum variable
    *  2. Split the world communicator into level communicators . Collect the roots of each level and make the reduce
    *     communicator
    *  3. Create level instance and compute.
    *  4. Compute sum with mpi reduce and get final mean of QoI.
    */
    template<class T>
    result_type level_run(std::vector<T>& Ml, std::vector<int>& Pl, std::vector<int>& Dl) {

        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        result_type fullsum, result;

        boost::mpi::communicator level_communicator = world_.split(level_);
        boost::mpi::communicator reduce_communicator = world_.split(tools::is_level_root(Pl, Dl, world_.rank()));

        Level<FunctionType, SamplerType> MLMC_level(integrand_, sample_generator_, level_, Ml[level_],
                                                    Pl[level_], Dl[level_], level_communicator);

        result = MLMC_level.compute();

        if (tools::is_level_root(Pl, Dl, world_.rank())) tools::reduce_helper(reduce_communicator, result, fullsum, std::plus<result_type>(), 0);

        return fullsum;
    }

    /**
     * @brief solves the parallel  integrand_ . Called by the user of the library
     * @param Ml  Vector with size equalling number of levels. Each component indicates the number of samples to be
     *            computed at this level.
     * @param Pl  Vector with size equalling number of levels. Each component indicates the number of samples computed
     *            simultaneously at every level
     * @param Dl  Vector with size equalling
     * @returns
     *
     * 1. Check if required number of cpus are available
     * 2. compute color
     * 3. compute level by calling level_compute_parallel()
     * 4. return result
     */
    template<class T>
    result_type run(std::vector<T>& Ml, std::vector<int>& Pl,
                      std::vector<int> Dl = std::vector<int>())
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

        // resize Dl if default arg used
        if (Dl.empty()) Dl.resize(Pl.size(), 1);

        // throws if there is a problem with Ml, Pl, Dl.
        tools::check_arguments(Ml, Pl, Dl);

        // change Pl if too large
        for (int i=0; i<Pl.size(); ++i) Pl[i] = std::min<long>(Pl[i],Ml[i]);

        int N = tools::num_required_processors(Ml, Pl, Dl);
        // throw if not enough CPUs
        if (world_.size() < N) {
            std::stringstream err;
            err << "Not enough CPUs! is: " << world_.size()
                << ", should be: " << N;
            throw std::runtime_error(err.str());
        }
        // if too many, warn and split off those that are too many
        if (world_.size() > N) {
            //if (world_.rank() == 0)
            //    std::cerr << "WARNING: too many CPUs. is: " << world_.size()
            //              << ", should be: " << N
            //              << std::endl;
            bool continue_sim = world_.rank() < N;
            world_ = world_.split(world_.rank() < N);
            if (!continue_sim) return result_type();
        }

        // start computation
        this->level_ = tools::get_level(Pl, Dl, world_.rank());

        auto result = level_run(Ml, Pl, Dl);
        return result;
    }
};

/**
 * Wrapper function that executes MLMC and returns the result.
 * @param integrand function to approximate
 * @param sampler function returning random samples
 * @param Ml vector of length L+1 containing number of samples per level
 * @param Pl vector of length L+1 containing number of samplesolvers per level
 * @param Dl vector of length L+1 containing number processes per samplesolver on each level
 * @param comm communicator to use for the simulation (defaults to world)
 */
template <class function_type, class sampler_type, class T>
typename tools::return_type<function_type, decltype(std::declval<sampler_type>()(0))>::type
    generic_MLMC(function_type integrand, sampler_type sampler, std::vector<T> Ml, std::vector<int> Pl,
                 std::vector<int> Dl, boost::mpi::communicator comm = boost::mpi::communicator()) {
    genericMLMC<function_type, sampler_type> gmlmc(integrand, sampler, comm);
    return gmlmc.run(Ml, Pl, Dl);
}

}  // end of namespace MLMC


#endif /* end of include guard: GENERICMLMC__GENERICMLMC_HPP */
