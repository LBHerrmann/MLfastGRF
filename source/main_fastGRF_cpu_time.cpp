//! @file   main_fastGRF_cpu_time.hpp
//! @author Lukas Herrmann
//! @date   2020
//
// ATTENTION: IN HEADER fe/boundary_dof_marker.hpp the two types
// interiorFESpace_t;
// boundaryFESpace_t;
// need to be make public
//
// system includes -------------------------------------------------------------
#include <string>
#include <iostream>
#include <memory>
#include <getopt.h>
#include <random>
#include <chrono>
#include <ctime>
#include <cmath>


#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/MatrixFunctions>


// Own includes
#include "precomputation.cpp"
#include "compute_fastGRF_cpu_time.cpp"


//-----------------------------------------------------------------------

// parameters with default values
int numRefines_;
int rep;
std::string basename_;
double beta;
double kappa1 = 1.0;
double kappa2 = 1.0;

//------------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
    //read parameters
    const char* const short_opts = "m:r:n:b:k:K";
    const option long_opts[] = {
        {"mesh", required_argument, nullptr, 'm'},
        {"rep", required_argument, nullptr, 'r'},
        {"num", required_argument, nullptr, 'n'},
        {"beta",  required_argument, nullptr, 'b'},
        {"kappa1",  optional_argument, nullptr, 'k'},
        {"kappa2",  optional_argument, nullptr, 'K'},
        {nullptr, no_argument, nullptr, 0}
    };
    while(true)
    {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
        if (-1 == opt) {
            break;
        }
        switch(opt)
        {
        case 'm':
            basename_ = std::string(optarg);
            break;
        case 'r':
            rep = std::stoi(optarg);
            break;
        case 'n':
            numRefines_ = std::stoi(optarg);
            break;
        case 'b':
            beta = std::stod(optarg);
            break;
        case 'k':
            kappa1 = std::stod(optarg);
            break;
        case 'K':
            kappa2 = std::stod(optarg);
            break;

        }
    }



    typedef Eigen::SparseMatrix< double > matrix_t;
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_t;
    std::vector< typename std::shared_ptr<matrix_t> > Ih_levels;
    std::vector< typename std::shared_ptr<matrix_t> > Ah_levels;
    std::vector< typename std::shared_ptr<matrix_t> > Mh_levels;

    std::vector< int > dimFEspace(numRefines_ + 1);


    // parse the command line
    const std::string basename = "source/" + basename_;
    const int numRefines =  numRefines_;

    // start pre-computations
    precomputation(basename,
                   numRefines_,
                   beta,
                   kappa1,
                   kappa2,
                   Ih_levels,
                   Ah_levels,
                   Mh_levels,
                   dimFEspace);

    std::vector<double> cpu_time(numRefines);

    // start to compute the cpu time for fastGRF
    compute_fastGRF_cpu_time(beta,
                             rep,
                             numRefines,
                             Ih_levels,
                             Mh_levels,
                             Ah_levels,
                             cpu_time);

    //============================================================================
    // EXPORTER SECTION
    //============================================================================





    std::ofstream outFile_dimFE( "results/cpu_time/" + basename_ + "_dimFEspace.txt" );

    for (const auto &e : dimFEspace) outFile_dimFE << e << "\n";

    std::ofstream outFile_cpu_time( "results/cpu_time/" + basename_ + "_cpu_time.txt" );

    for (const auto &e : cpu_time) outFile_cpu_time << e << "\n";






    //============================================================================
    // A FINAL STATEMENT
    //============================================================================




    std::cout << std::endl << "Program terminated normally!" << std::endl;

    return 0;
}

