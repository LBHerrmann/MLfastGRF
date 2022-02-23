//! @file   main_fastGRF_L2L2_error.cpp
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
#include <cmath>

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/MatrixFunctions>

// genericMLMC includes --------------------------------------------------------
#include <genericMLMC/tools/vector_ops.hpp>
#include <genericMLMC/genericMLMC.hpp>
#include <jsoncons/json.hpp>


// Own includes
#include "lambda_min_max_SPD.hpp"
#include "parametric_integrand_bpx_cg_L2L2_error.hpp"
#include "sampler.hpp"
#include "precomputation.cpp"

// -----------------------------------------------------------------------------

// parameters with default values
int numRefines_;
int Ncpu_;
std::string basename_;
double beta;
double kappa1;
double kappa2;

//-----------------------------------------------------------------------

#define pout \
    if (world.rank() == 0) std::cout

//------------------------------------------------------------------------------
int main( int argc, char* argv[] )
{

    boost::mpi::environment env;
    boost::mpi::communicator world;

    //read parameters
    const char* const short_opts = "m:p:n:b:k:K";
    const option long_opts[] = {
        {"mesh", required_argument, nullptr, 'm'},
        {"ncpu", required_argument, nullptr, 'p'},
        {"num", required_argument, nullptr, 'n'},
        {"beta",  required_argument, nullptr, 'b'},
        {"kappa1",  required_argument, nullptr, 'k'},
        {"kappa2",  required_argument, nullptr, 'K'},
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
            //std::cout << "Mesh set to: " << basename_ << std::endl;
            break;
        case 'p':
            Ncpu_ = std::stoi(optarg);
            break;
        case 'n':
            numRefines_ = std::stoi(optarg);
            //std::cout << "NumRef set to: " << numRefines_ << std::endl;
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




    //========================================
    // compute smallest and largest eigenvalue of mass matrix
    //========================================



    const double eps = std::numeric_limits<double>::epsilon();

    const double threshold = 100.0 * eps;

    int power_steps = 20;


    matrix_t Mass = *(Mh_levels[numRefines]);

    vector_t ones(Mass.rows());
    ones.setOnes();

    std::cout<<"starting lambda max computation"<<std::endl;

    lambda_min_max lambda_min_max_(threshold,power_steps);

    double lambda_max = lambda_min_max_.lambda_max(Mass, ones   );

    std::cout<< "lambda max = " << lambda_max <<std::endl;

    double lambda_min = lambda_min_max_.lambda_min(Mass, ones   );

    std::cout<< "lambda min = " << lambda_min <<std::endl;




    //===========================================================================
    //
    // SETUP THE RANDOM NUMBER GENERATOR TO GENERATE SEEDS
    //
    //===========================================================================

    //random number generator for the seed
    typedef std::mt19937 base_generator_t;
    base_generator_t generator_int;
    std::uniform_int_distribution<int> dist_int(1,std::numeric_limits<int>::max( ) );
    generator_int.seed( 5489 + world.rank() );

    int Ncpu = Ncpu_;


    bool iter = false;
    if (beta > 1 && beta < 2) {
        iter = true;
    }


    parametric_integrand_bpx_pcg_L2L2_err GRF_bpx_pcg_L2L2_err( numRefines,
            beta,
            lambda_max * 1.1,
            lambda_min / 1.1,
            iter);


    GRF_bpx_pcg_L2L2_err.setup(Ih_levels, Ah_levels, Mh_levels);


    const unsigned seed_level = dist_int( generator_int );


    samplerclass sampler( Mass.rows(), seed_level);





    std::vector< double > newresult( numRefines + 1  );
    std::fill(newresult.begin(), newresult.end(), 0.0);

    std::vector<int> Ml{60};  // number of samples per level
    std::vector<int> Pl{Ncpu};
    std::vector<int> Dl{1};

    std::cout << "start GRF computation" << std::endl;

    newresult = MLMC::generic_MLMC(
    [&](std::vector< double > spl, int l) {
        return GRF_bpx_pcg_L2L2_err(spl,0);
    },
    sampler, Ml, Pl, Dl);





    //============================================================================
    // EXPORTER SECTION
    //============================================================================
    if (world.rank() == 0)
    {

        std::string beta_str = std::to_string(beta);
        beta_str = beta_str.substr(0, beta_str.find('.') + 3);
        std::string kappa1_str = std::to_string(kappa1);
        kappa1_str = kappa1_str.substr(0, kappa1_str.find('.'));
        std::string kappa2_str = std::to_string(kappa2);
        kappa2_str = kappa2_str.substr(0, kappa2_str.find('.'));

        std::ofstream outFile_L2L2( "results/beta_" + beta_str
                                    + "/" + basename_ + "_L2L2_error_" + kappa1_str + "_" + kappa2_str + ".txt" );

        for (const auto &e : newresult) outFile_L2L2 << e << "\n";

        std::ofstream outFile_dimFE( "results/beta_" + beta_str
                                     + "/" + basename_ + "_dimFEspace.txt" );

        for (const auto &e : dimFEspace) outFile_dimFE << e << "\n";

    }


    //============================================================================
    // A FINAL STATEMENT
    //============================================================================




    std::cout << std::endl << "Program terminated normally!" << std::endl;

    return 0;
}

