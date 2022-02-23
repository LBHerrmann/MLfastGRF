//! @file   parametric_integrand_bpx_pcg_L2L2_err.hpp
//! @author Lukas Herrmann
//! @date   2020

#ifndef PARAMETRIC_INTEGRAND_BPX_PCG_L2L2_ERR_HPP
#define PARAMETRIC_INTEGRAN_BPX_PCG_L2L2_ERR_HPP


// system includes -------------------------------------------------------------
#include <string>
#include <iostream>
#include <memory>

#include <cmath>

#include <Eigen/IterativeLinearSolvers>

// eth includes ----------------------------------------------------------------
#include <input_interface/input_interface.hpp>
#include <grid_utils/grid_view_factory.hpp>
#include <utils/sparsestream.hpp>

// betl2 includes --------------------------------------------------------------
#include <cmdl_parser/cmdl_parser.hpp>
#include <gmsh_input/gmsh_input.hpp>
#include <gmsh_input/grid_elements_identifier.hpp>
#include <grid/volume2d_grid.hpp>
#include <grid/grid_view.hpp>

#include <fe/febasis.hpp>
#include <fem_operator/finite_element.hpp>
#include <fem_operator/bilinear_form.hpp>
#include <fem_operator/linear_form.hpp>
#include <fem_operator/intersection_bilinear_form.hpp>
#include <fem_operator/intersection_linear_form.hpp>

#include <fe/fe_enumerators.hpp>
#include <fe/dof_handler.hpp>
#include <fe/boundary_dof_marker.hpp>
#include <fe/intersections_dof_marker.hpp>
#include <fe/constrained_fespace.hpp>

#include <quadrature/quadrature_list.hpp>
#include <integration/integrator.hpp>
#include <integration/linear_integrator.hpp>
#include <integration/intersection_integrator.hpp>
#include <integration/intersection_linear_integrator.hpp>

#include <material/trivial_material.hpp>
#include <functional/analytical_grid_function.hpp>
#include <functional/dof_interpolator.hpp>
#include <functional/grid_function_operations.hpp>

#include <functional/interpolation_grid_function.hpp>
#include <vtu_exporter/vtu_exporter.hpp>


// sparse operators ------------------------------------------------------------
#include <sparse_operators/identity_operator.hpp>
#include <multilevel/multilevel_operators.hpp>

// boost includes --------------------------------------------------------------
#include <boost/lexical_cast.hpp>

// own includes ----------------------------------------------------------------
#include "sqrt_matrix.hpp"
#include "bpx_preconditioner.hpp"

using namespace betl2;


//------------------------------------------------------------------------------

namespace betl2 {
class parametric_integrand_bpx_pcg_L2L2_err
{
private:
    //============================================================================
    //
    // define some BETL types
    //
    //============================================================================

    typedef double numeric_t;
    typedef Eigen::Matrix<numeric_t,Eigen::Dynamic,1> vector_t;
    typedef Eigen::Matrix<numeric_t,Eigen::Dynamic,Eigen::Dynamic> matrix_full_t;
    typedef Eigen::SparseMatrix<numeric_t> matrix_t;

    typedef std::vector< typename std::shared_ptr<matrix_t> > smart_ptr_matrix_t;



    const numeric_t value_of_PI = 3.141592653589793238462643383279;


    //class members
    unsigned numRefines_;
    const smart_ptr_matrix_t* Prolongation_;
    const smart_ptr_matrix_t* System_matrices_;
    const smart_ptr_matrix_t* Mass_matrices_;
    Eigen::SparseLU< matrix_t > direct_solver_;


    // parameters
    numeric_t beta_;
    numeric_t lambda_max_;
    numeric_t lambda_min_;

    bool iter_;




public:
    parametric_integrand_bpx_pcg_L2L2_err(   	 const unsigned numRefines,
            numeric_t beta,
            numeric_t lambda_max,
            numeric_t lambda_min,
            bool iter )
        :numRefines_( numRefines ),
         beta_( beta ),
         lambda_max_( lambda_max ),
         lambda_min_( lambda_min ),
         iter_( iter ),
         Prolongation_( nullptr ),
         System_matrices_( nullptr ),
         Mass_matrices_( nullptr )
    {

    }

    //destructor
    ~parametric_integrand_bpx_pcg_L2L2_err( )
    {
        if( Prolongation_ != nullptr ) delete Prolongation_;
        if( System_matrices_ != nullptr ) delete System_matrices_;
        if( Mass_matrices_ != nullptr ) delete Mass_matrices_;
    }

    void setup( const smart_ptr_matrix_t& Prolongation,
                const smart_ptr_matrix_t& System_matrices,
                const smart_ptr_matrix_t& Mass_matrices)
    {
        Prolongation_ = new smart_ptr_matrix_t( Prolongation );
        System_matrices_ = new smart_ptr_matrix_t( System_matrices );
        Mass_matrices_ = new smart_ptr_matrix_t( Mass_matrices );

    }



    std::vector< double >  operator()( std::vector< numeric_t > y, int j)

    {

        const int UPLO = Eigen::Lower;
        typedef BPXPreconditioner< numeric_t > precond_t;
        typedef Eigen::ConjugateGradient< Eigen::SparseMatrix<numeric_t>, UPLO, precond_t > solver_t;






        smart_ptr_matrix_t Mass;



        smart_ptr_matrix_t Prolongation;


        //assign the prolongation, mass
        for(int i=0; i<=numRefines_; i++)
        {


            matrix_t Mh = (*((*Mass_matrices_)[i]));
            std::shared_ptr<matrix_t> Mh_level_ptr( new matrix_t( Mh ) );
            Mass.push_back( Mh_level_ptr );

            if(i<numRefines_)
            {
                matrix_t Ih = (*((*Prolongation_)[i]));
                std::shared_ptr<matrix_t> Ih_level_ptr( new matrix_t( Ih ) );
                Prolongation.push_back( Ih_level_ptr );
            }

        }



        //==========================================
        // compute reference solution using sqrt M
        //==========================================


        Eigen::Map<vector_t> Y_vec(&y[0],y.size());

        sqrt_matrix sqrt_matrix_( lambda_min_, lambda_max_);



        vector_t RHS_white_noise = sqrt_matrix_.apply_sqrt(20, *(Mass[numRefines_]), Y_vec);
        //vector_t RHS_white_noise = (*(Mass[numRefines_]) ) * Y_vec;

        std::vector<vector_t> GRF_levels(numRefines_);

        //step1 L2 project RHS_white_noise to be able to compare realizations
        std::vector<vector_t> RHS_vector(numRefines_+1);
        RHS_vector[numRefines_] = RHS_white_noise;

        // perform L2 projection
        for(int i=numRefines_-1; i >= 0; i--)
        {
            RHS_vector[i] = ((*(Prolongation[i])).transpose() )  * (RHS_vector[i+1]);



        }

        //start to compute for ref_sol





        //==========================================
        // sinc quadrature for ref solution
        //==========================================

        numeric_t rate = 2.0 * (beta_ + ((int)iter_ ))- 1.0;
        int K = int( ((rate/4.0/(1-beta_)) * std::log( ((double) RHS_white_noise.rows()) )) * ((rate/4.0/(1-beta_)) * std::log( ((double) RHS_white_noise.rows()) ))  + 10 );

        numeric_t delta = 1.0 / std::sqrt((double)K );

        // solve for k=0
        const matrix_t& Ah = (*((*System_matrices_)[numRefines_]));
        const matrix_t& Mh = (*((*Mass_matrices_)[numRefines_]));


        if( iter_ )
        {

            direct_solver_.compute( Ah );
            vector_t w = direct_solver_.solve( RHS_white_noise );
            RHS_white_noise = Mh * w;



        }
        direct_solver_.analyzePattern( (Ah + Mh)   );
        direct_solver_.factorize( (Ah + Mh)   );
        vector_t GRF_ref = direct_solver_.solve( RHS_white_noise );


        for(int k=1; k<=K; k++)
        {
            // for k
            direct_solver_.factorize( (std::exp( - 2.0 * ((double)k) * delta * (1.0 - beta_)) * Ah
                                       + std::exp(  2.0 * ((double)k) * delta * beta_) * Mh  ) );
            // update with increment
            GRF_ref +=   (direct_solver_.solve( RHS_white_noise ));

            //for -k
            direct_solver_.factorize( (std::exp(  2.0 * ((double)k) * delta * (1.0 - beta_)) * Ah
                                       + std::exp(   - 2.0 * ((double)k) * delta * beta_) * Mh  ) );
            // update with increment
            GRF_ref +=   (direct_solver_.solve( RHS_white_noise ));

        }

        GRF_ref = delta * GRF_ref;


        //=======================================
        // start to solve on levels with BPX PCG
        //=======================================



        int K_level = int( ((rate/4.0/(1-beta_)) * std::log( ((double) (RHS_vector[0]).rows())) ) * ((rate/4.0/(1-beta_)) * std::log( ((double) (RHS_vector[0]).rows()) )) )   + 1 ;
        //int K_level = 1;

        //on level 0 apply direct solver
        numeric_t delta_level = 1.0 / std::sqrt((double)K_level );

        // solve for k=0
        const matrix_t& Ah_level = (*((*System_matrices_)[0]));
        const matrix_t& Mh_level = (*((*Mass_matrices_)[0]));

        if( iter_ )
        {

            direct_solver_.compute( Ah_level );
            vector_t w = direct_solver_.solve( (RHS_vector[0]) );
            (RHS_vector[0]) = Mh_level * w;



        }


        direct_solver_.compute( (Ah_level + Mh_level)   );
        vector_t GRF_level = direct_solver_.solve( (RHS_vector[0]) );

        for(int k=1; k<=K_level; k++)
        {

            // for k
            direct_solver_.compute( (std::exp( - 2.0 * ((double)k) * delta_level * (1.0 - beta_) ) * Ah_level
                                     + std::exp(  2.0 * ((double)k) * delta_level *  beta_ ) * Mh_level  ) );
            // update with increment
            GRF_level +=   (direct_solver_.solve( (RHS_vector[0]) ));

            //for -k
            direct_solver_.compute( ((std::exp(  2.0 * ((double)k) * delta_level * (1.0 - beta_) )) * Ah_level
                                     + (std::exp(  - 2.0 * ((double)k) * delta_level *  beta_ ) ) * Mh_level  ) );
            // update with increment
            GRF_level +=   (direct_solver_.solve( (RHS_vector[0]) ));

        }

        GRF_levels[0] = delta_level * GRF_level;



        // loop over levels; actually need BPX PCG here
        for(int ell=1; ell< numRefines_; ell++)
        {
            //=======================
            //start sinc quad for k=-K,...,K
            //=======================

            vector_t GRF_level( (*(Prolongation[ell])).cols() );
            GRF_level.setZero();

            int K_level =  int( ((rate/4.0/(1-beta_)) * std::log( ((double) (RHS_vector[ell]).rows())) ) * ((rate/4.0/(1-beta_)) * std::log( ((double) (RHS_vector[ell]).rows()) )) )  + 1 ;
            //int K_level = 1;
            delta_level = 1.0 / (std::sqrt( ((double)K_level)  ) );


            if( iter_ )
            {

                smart_ptr_matrix_t System;
                // get the right system matrices for BPX
                for(int i=0; i<=ell; i++)
                {
                    matrix_t Ah_level = (*((*System_matrices_)[i]));
                    std::shared_ptr<matrix_t> Sh_level_ptr( new matrix_t(
                            Ah_level  )  );
                    System.push_back( Sh_level_ptr );

                }

                // solve with BPX PCG for k == 0
                solver_t solver_bpx_pcg;
                auto& preconditioner = solver_bpx_pcg.preconditioner();

                preconditioner.setup( Prolongation, System, ell+1 );
                solver_bpx_pcg.setMaxIterations( 100 );
                solver_bpx_pcg.setTolerance( 1.e-12 );


                solver_bpx_pcg.compute( (*(System[ell])) );
                vector_t w =  (solver_bpx_pcg.solve( (RHS_vector[ell]) ));
                (RHS_vector[ell]) = (*(Mass[ell])) * w;




            }


            // perform sinc for k=1,...,K

            for(int k=-K_level; k<=K_level; k++)
            {

                smart_ptr_matrix_t System;
                // get the right system matrices for BPX
                for(int i=0; i<=ell; i++)
                {
                    const matrix_t& Ah_level = (*((*System_matrices_)[i]));

                    std::shared_ptr<matrix_t> Sh_level_ptr( new matrix_t(
                            (std::exp(-2.0 * ((double) k) * delta_level * (1.0 - beta_) )  * Ah_level
                             + (std::exp(2.0 * ((double) k) * delta_level * beta_ )) * (*(Mass[i])))  ) );
                    System.push_back( Sh_level_ptr );

                }

                // solve with BPX PCG for k == 0
                solver_t solver_bpx_pcg;
                auto& preconditioner = solver_bpx_pcg.preconditioner();

                preconditioner.setup( Prolongation, System, ell+1 );

                solver_bpx_pcg.setMaxIterations( 100 );
                solver_bpx_pcg.setTolerance( 1.e-12 );


                solver_bpx_pcg.compute( (*(System[ell])) );
                GRF_level +=  (solver_bpx_pcg.solve( (RHS_vector[ell]) ));

            }

            GRF_levels[ell] = delta_level * GRF_level;




        }

        //interpolate the GRFs to leaf level
        for(int ell=0; ell<numRefines_; ell++)
        {
            for(int i=0; i<=ell; i++)
            {
                GRF_levels[i] =  (*(Prolongation[ell])) * (GRF_levels[i]);
            }

        }

        // compute L2 error with mass matrix
        std::vector<double> L2_error_vector;

        numeric_t GRF_ref_norm2 = (GRF_ref.transpose()) * (*(Mass[numRefines_])) * GRF_ref;

        for(int ell=0; ell<numRefines_; ell++)
        {
            vector_t res = GRF_ref - GRF_levels[ell];
            numeric_t L2_err = ((res.transpose()) * (*(Mass[numRefines_]))* res ) ;
            //numeric_t L2_err = ((res.transpose()) * Ah * res ) ;
            L2_error_vector.push_back( L2_err );

        }

        L2_error_vector.push_back( GRF_ref_norm2 );

        return L2_error_vector;






    }


};//end of class parametric_integrand

}// end namespace betl2

#endif
