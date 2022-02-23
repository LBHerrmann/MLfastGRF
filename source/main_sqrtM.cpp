//! @file   main_sqrtM.hpp
//! @author Lukas Herrmann
//! @date   2020
//
// ATTENTION: IN HEADER fe/boundary_dof_marker.hpp the two types
// interiorFESpace_t;
// boundaryFESpace_t;
// need to be make public

// system includes -------------------------------------------------------------
#include <string>
#include <iostream>
#include <memory>
#include <getopt.h>
#include <random>
#include <cmath>

// Own includes
#include "compute_sqrtM_err.cpp"

//-----------------------------------------------------------------------

//parameters
int numRefines_ = 0;
std::string basename_;
//------------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
    //read parameters
    const char* const short_opts = "m:n";
    const option long_opts[] = {
        {"mesh", required_argument, nullptr, 'm'},
        {"num", required_argument, nullptr, 'n'},
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
        case 'n':
            numRefines_ = std::stoi(optarg);
            break;
        }
    }

    const std::string basename = "source/" + basename_;

    std::vector<double> sqrt_err;

    compute_sqrtM_err(basename, numRefines_, sqrt_err);



    //============================================================================
    // A FINAL STATEMENT
    //============================================================================


    std::string numRefines_str = std::to_string(numRefines_);

    std::ofstream outFile_sqrt_err( "results/sqrt_err/sqrt_err_" + numRefines_str + ".txt" );
    // the important part
    for (const auto &e : sqrt_err) outFile_sqrt_err << e << "\n";


    std::cout << std::endl << "Program terminated normally!" << std::endl;

    return 0;
}

