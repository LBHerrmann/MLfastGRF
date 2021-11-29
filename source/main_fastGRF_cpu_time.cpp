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

#include <chrono>
#include <ctime>



#include <cmath>

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/MatrixFunctions>

// eth includes ----------------------------------------------------------------
#include <input_interface/input_interface.hpp>
#include <grid_utils/grid_view_factory.hpp>
#include <utils/sparsestream.hpp>

// betl2 includes --------------------------------------------------------------
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



// Own includes
#include "simple_functions.hpp"
#include "bpx_preconditioner.hpp"
#include "lambda_min_max_SPD.hpp"
#include "sqrt_matrix.hpp"


// -----------------------------------------------------------------------------


using namespace betl2;
namespace big = betl2::input::gmsh;

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
        if (-1 == opt){break;}
        switch(opt)
        {
                case 'm': 
                        basename_ = std::string(optarg);
                        //std::cout << "Mesh set to: " << basename_ << std::endl;
                        break;
                case 'r':
                        rep = std::stoi(optarg);
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
	
  // parse the command line
  const std::string basename = "source/" + basename_;

  // create input
  std::cout << "Input from: " << basename << ".msh" << std::endl;
  big::Input input( basename );
  std::cout << input << std::endl;
    
  // wrap input interface around the given input
  typedef betl2::input::InputInterface< big::Input > inpInterface_t;
  inpInterface_t inpInterface( input );
  
  // instantiate the grid implementation
  typedef betl2::volume2dGrid::hybrid::Grid grid_t;
  typedef std::shared_ptr< eth::grid::Grid<grid_t::gridTraits_t> > grid_ptr_t;
  grid_ptr_t grid_ptr( new grid_t( inpInterface ) );
  {
    typedef betl2::GeometryMapper<grid_t::gridTraits_t> geomBaseMapper_t;
    typedef betl2::IdentityMapper<grid_t::gridTraits_t> geomMapper_t;
    typedef std::shared_ptr<geomBaseMapper_t> baseMapper_ptr_t;
    typedef std::shared_ptr<geomMapper_t    > geomMapper_ptr_t;
  
    auto& refinement = grid_ptr -> refinement( );
    baseMapper_ptr_t geomMapper( new geomMapper_t );
    const auto elementCollection = grid_ptr -> leafEntities<0>();
    refinement.registerMapping( geomMapper, elementCollection );
  }
  
  //set the number of refinements
  const int numRefines = numRefines_;
  grid_ptr -> globalRefine( numRefines );
  
  
  // create a grid view factory
  typedef eth::grids::utils::GridViewFactory< grid_t, eth::grid::GridViewTypes::LeafView > surfaceGridFactory_t;
  surfaceGridFactory_t surfaceGridFactory( grid_ptr );

  const auto gridView = surfaceGridFactory.getView();
  
  const int numNodes    = gridView.size( 2 );
  const int numEdges    = gridView.size( 1 );
  const int numElements = gridView.size( 0 );
  
  std::cout << "Leaf:" << std::endl
            << "\t#elements = " << numElements << std::endl
            << "\t#edges    = " << numEdges    << std::endl
            << "\t#nodes    = " << numNodes    << std::endl;
  
 
  
  //============================================================================
  // SETUP OF FEBASIS AND FESPACE, INTEGRATOR, BILINEAR-FORM
  //============================================================================
  // define a constant febasis
  typedef fe::FEBasis< fe::Linear, fe::FEBasisType::Lagrange > febasis_t;
  // and some test/trial basis functions:
  typedef fem::FiniteElement< febasis_t, fe::FEDiff::Identity,
                              febasis_t, fe::FEDiff::Identity,
                              1, 2 > femMass_t;               //for int_{\Omega}(U*V)
  
  typedef fem::FiniteElement< febasis_t, fe::FEDiff::Grad,
                              febasis_t, fe::FEDiff::Grad,
                              2, 2 > femStiff_t;              //for int_{\Omega}(gradU A gradV)
  

  // define dofhandler type for the surface grid
  typedef betl2::fe::DofHandler< febasis_t, fe::FESContinuity::Continuous, surfaceGridFactory_t  > surfDH_t;

  // instantiate dofhander for surface grid
  surfDH_t  surfDH;

  // distribute the degrees of freedom
  surfDH.distributeDofs ( surfaceGridFactory );

  std::cout << "Number of created dofs for surface (unconstrained) = " << surfDH.numDofs() << std::endl;
  
  fe::BoundaryDofMarker< surfDH_t::fespace_t > marker( surfDH.fespace() );
  marker.mark( surfaceGridFactory ); // this call modifies the fespace!!!
  
  // extract the constrained space
  typedef fe::BoundaryDofMarker< surfDH_t::fespace_t >::interiorFESpace_t constrained_space_t;
  const constrained_space_t& constrained_space = marker.interiorFESpace( );
  
  // extract a finite element space defined on the boundary 
  typedef fe::BoundaryDofMarker< surfDH_t::fespace_t >::boundaryFESpace_t boundary_space_t;
  const boundary_space_t& boundary_space = marker.boundaryFESpace( );
  
  std::cout << "Number of created interior-dofs = " << constrained_space.numDofs() << std::endl;  
  std::cout << "Number of created boundary-dofs = " << boundary_space.numDofs()    << std::endl;
  
  //============================================================================
  // SETUP OF INTEGRATORS AND BILINEAR-FORMS
  //============================================================================
  //INTEGRATORS: ---------------------------------------------------------------
  // Quadrature rules for element type (surface):
  typedef QuadRule< eth::base::RefElType::TRIA,3 > tria_t;
  typedef QuadRule< eth::base::RefElType::QUAD,4 > quad_t;
  // Quadrature rules for element type (edge):
  typedef QuadRule< eth::base::RefElType::SEGMENT,2 > line_t;
  // subsume quadrature rules in typelists
  typedef QuadRuleList< tria_t, quad_t > surfQuadrules_t;
  typedef QuadRuleList< line_t > lineQuadrules_t;

  //  Actual integrators:
  typedef double numeric_t;
  typedef fem::Integrator<              numeric_t, femStiff_t,    surfQuadrules_t >   stiffIntegrator_t;
  typedef fem::Integrator<              numeric_t, femMass_t,     surfQuadrules_t >   massIntegrator_t;

  //FEM OPERATORS: -------------------------------------------------------------
  typedef fem::BilinearForm<              stiffIntegrator_t     >     stiffnessBilinearForm_t;
  typedef fem::BilinearForm<              massIntegrator_t      >     massBilinearForm_t;

  

   

  
  
 
  //============================================================================
  // PreCOMPUTE ON REFINEMENT LEVELS
  //============================================================================
  
  typedef eth::grids::utils::GridViewFactory< grid_t, eth::grid::GridViewTypes::LevelView > surfaceGridFactory_level_t;
  
  // define dofhandler type for the surface grid on levels
  typedef betl2::fe::DofHandler< febasis_t, fe::FESContinuity::Continuous, surfaceGridFactory_level_t  > surfDH_level_t;
  
  typedef Eigen::SparseMatrix<numeric_t> matrix_t;
  typedef Eigen::Matrix< numeric_t, Eigen::Dynamic, 1 > vector_t;
  std::vector< typename std::shared_ptr<matrix_t> > Ih_levels;
  std::vector< typename std::shared_ptr<matrix_t> > Ah_levels;
  std::vector< typename std::shared_ptr<matrix_t> > Mh_levels;
  std::vector< vector_t> sol_vector( numRefines +2 );
  std::vector< int > dimFEspace( numRefines +1 );
  
  dimFEspace[numRefines] = constrained_space.numDofs();
  
  for (int i=numRefines;i>=1;i--)
    { 
		//================================
		//compute the FE meshes on level
		//================================
		surfaceGridFactory_level_t surfaceGridFactory_level( grid_ptr, i );
		
		// instantiate dofhander for surface grid
		surfDH_level_t  surfDH_level;

		// distribute the degrees of freedom
		surfDH_level.distributeDofs ( surfaceGridFactory_level );

		  
		fe::BoundaryDofMarker< surfDH_level_t::fespace_t > marker_level( surfDH_level.fespace() );
		marker_level.mark( surfaceGridFactory_level ); // this call modifies the fespace!!!
		
		 
		// extract the constrained space
		typedef fe::BoundaryDofMarker< surfDH_level_t::fespace_t >::interiorFESpace_t constrained_space_level_t;
		const constrained_space_level_t& constrained_space_level = marker_level.interiorFESpace( );
		dimFEspace[numRefines-i] = constrained_space_level.numDofs();
		
	    // approx mesh width
	    numeric_t h_level = 1/sqrt( double(dimFEspace[numRefines-i]) ); 
		  
		// extract a finite element space defined on the boundary 
		typedef fe::BoundaryDofMarker< surfDH_level_t::fespace_t >::boundaryFESpace_t boundary_space_level_t;
		const boundary_space_level_t& boundary_space_level = marker_level.boundaryFESpace( );
		
		//==================================================================
		// compute mass matrices
		//==================================================================
		
		//Compute Mass Matrix
		
			fem::TrivialMaterial<surfaceGridFactory_level_t>      gamma( surfaceGridFactory_level, 1.0 );
			massBilinearForm_t M_level;
			M_level.compute( constrained_space_level, gamma );
			M_level.make_sparse();
			
			std::shared_ptr<matrix_t> Mh_level_ptr( new matrix_t( M_level.matrix() ) );
		    Mh_levels.push_back( Mh_level_ptr );
		
		//==================================================================
		// compute stiffness matrices
		//==================================================================
		
		//Compute Stiffness Matrix
		
			fem::TrivialMaterial<surfaceGridFactory_level_t>      gamma_0( surfaceGridFactory_level, 1.0 );
			stiffnessBilinearForm_t A_level;
			A_level.compute( constrained_space_level, gamma_0 );
			A_level.make_sparse();
			
	    //compute reaction term 
	        fem::ReactionTerm<surfaceGridFactory_level_t>      kappa( surfaceGridFactory_level, kappa1, kappa2 );
			massBilinearForm_t M_react_level;
			M_react_level.compute( constrained_space_level, kappa);
			M_react_level.make_sparse();
			
			std::shared_ptr<matrix_t> Ah_level_ptr( new matrix_t( ( A_level.matrix() + M_react_level.matrix() ) ) );
		    Ah_levels.push_back( Ah_level_ptr );
		    
		
		
		
		//==================================================================
		// compute prolongation operators
		//==================================================================
		
		
		
		if ( i==1 )
		{
		  typedef betl2::multilevel::InterpolationOperator< constrained_space_t, 
												   constrained_space_level_t > interpolation_op_t;
		  interpolation_op_t interpolation_op( constrained_space, constrained_space_level );
		  interpolation_op.compute( );
		  
		  

		  std::shared_ptr<matrix_t> Ih_level_ptr( new matrix_t( interpolation_op.matrix() ) );
		  Ih_levels.push_back( Ih_level_ptr );
		  
		}

        else
		{
		  surfaceGridFactory_level_t surfaceGridFactory_leaf( grid_ptr, i-1 );
		  surfDH_level_t  surfDH_leaf;
		  surfDH_leaf.distributeDofs ( surfaceGridFactory_leaf );
		  
		  // get constrained space on leaf
		  fe::BoundaryDofMarker< surfDH_level_t::fespace_t > marker_leaf( surfDH_leaf.fespace() );
		  marker_leaf.mark( surfaceGridFactory_leaf ); // this call modifies the fespace!!!
		  // extract the constrained space
		  const constrained_space_level_t& constrained_space_leaf = marker_leaf.interiorFESpace( );
		  
		  
		  
		  typedef betl2::multilevel::InterpolationOperator< constrained_space_level_t, 
													   constrained_space_level_t > interpolation_op_t;
		  interpolation_op_t interpolation_op( constrained_space_leaf, constrained_space_level );
		  interpolation_op.compute( );

		  std::shared_ptr<matrix_t> Ih_level_ptr( new matrix_t( interpolation_op.matrix() ) );
		  Ih_levels.push_back( Ih_level_ptr );

		}
		
		 
		
		
		
		
		
	}
	
    
    //Compute Mass Matrix
    fem::TrivialMaterial<surfaceGridFactory_t>      gamma( surfaceGridFactory, 1.0 );
    massBilinearForm_t M;
    M.compute( constrained_space, gamma );
    M.make_sparse();
    
    std::shared_ptr<matrix_t> Mh_level_ptr( new matrix_t( M.matrix() ) );
    Mh_levels.push_back( Mh_level_ptr );
	
  	//Compute Stiffness Matrix
    fem::TrivialMaterial<surfaceGridFactory_t>      gamma_0( surfaceGridFactory, 1.0 );
    stiffnessBilinearForm_t A;
    A.compute( constrained_space, gamma_0 );
    A.make_sparse();
    fem::ReactionTerm<surfaceGridFactory_t>      kappa( surfaceGridFactory, kappa1, kappa2 );
    massBilinearForm_t M_react;
    M_react.compute( constrained_space, kappa );
    M_react.make_sparse(); 
    
    std::shared_ptr<matrix_t> Ah_level_ptr( new matrix_t( (A.matrix() + M_react.matrix() ) ));
    Ah_levels.push_back( Ah_level_ptr );
	
 
  std::cout<<"precomputations completed"<<std::endl;
  
  
  //========================================
  // test sqrt M algorithm
  //======================================== 

  //random number generator for the seed
  typedef std::mt19937 base_generator_t;
  base_generator_t generator_norm;
  std::normal_distribution<double> dist_norm(0.0, 1.0);
  generator_norm.seed( 5489 );
  
  
  //test square root of mass matrix here
  
  //step 1 compute smallest and largest eigenvalue of mass matrix
  
  
  const double eps = std::numeric_limits<double>::epsilon();
  
  const double threshold = 100.0 * eps;
  
  int power_steps = 20;
  //int Reyleigh_steps = 20;
  
  
  matrix_t Mass = *(Mh_levels[numRefines]);
  
  vector_t ones(Mass.rows());
  ones.setOnes();
  
  std::cout<<"starting lambda max computation"<<std::endl;
  
  lambda_min_max lambda_min_max_(threshold,power_steps);
  
  double lambda_max = lambda_min_max_.lambda_max(Mass, ones   );
  
  std::cout<< "lambda max = " << lambda_max <<std::endl;
  
  double lambda_min = lambda_min_max_.lambda_min(Mass, ones   );
  
  std::cout<< "lambda min = " << lambda_min <<std::endl;
  
  
  
  int R = rep;
  
  const numeric_t value_of_PI = 3.141592653589793238462643383279;
  numeric_t beta_ = beta;
  
  numeric_t rate = 2.0 * beta_ - 1.0;
  
  const int UPLO = Eigen::Lower;
  typedef BPXPreconditioner< numeric_t > precond_t;
  typedef Eigen::ConjugateGradient< Eigen::SparseMatrix<numeric_t>, UPLO, precond_t > solver_t;
  
  std::vector< vector_t > GRF_levels( numRefines );
  
  std::vector<double> cpu_time(numRefines);
  for(int i=0; i<cpu_time.size(); i++)
  {
	  cpu_time[i] = 0.0;
  }
  
  
  int numRefines_ = numRefines;
  
  
  
  Eigen::SparseLU< matrix_t > direct_solver_;
  
  for(int r=0; r<R ; r++)
  {
	  std::cout<< "r = " << r << std::endl;
	 std::clock_t c_start = std::clock();
  
     //compute lambdas
     int power_steps = 20;
  
  
  
     
  
     vector_t ones((*(Mh_levels[0])).rows());
     ones.setOnes();
  
  
     lambda_min_max lambda_min_max_(1.0,power_steps);
  
     double lambda_max = lambda_min_max_.lambda_max((*(Mh_levels[0])), ones   );
  
  
     double lambda_min = lambda_min_max_.lambda_min((*(Mh_levels[0])), ones   );
  
     
     //compute RHS
     vector_t Y_0((*(Mh_levels[0])).rows());
     for(int j=0; j< Y_0.rows(); j++)
     {
        Y_0(j,0) = dist_norm(generator_norm);
   
     } 
     sqrt_matrix sqrt_matrix_( (lambda_min / 1.5), (lambda_max * 1.5));
     vector_t RHS_level0 = sqrt_matrix_.apply_sqrt(10, (*(Mh_levels[0])), Y_0);
     
	  
	  

	  int K_level = int( ((rate/4.0/(1-beta_)) * std::log( ((double) RHS_level0.rows())) ) * ((rate/4.0/(1-beta_)) * std::log( ((double) RHS_level0.rows()) )) ) + 1;
  	 
 	  //on level 0 apply direct solver
 	  numeric_t delta_level = 1.0 / std::sqrt((double)K_level );
	  
	  // solve for k=0
	  matrix_t Ah_level = *(Ah_levels[0]);
	  matrix_t Mh_level = *(Mh_levels[0]);
	  
	  direct_solver_.compute( (Ah_level + Mh_level)   );
	  vector_t GRF_level = direct_solver_.solve( (RHS_level0) );
	  
	  for(int k=1; k<=K_level; k++)
	  {
		  // for k
		  direct_solver_.compute( (std::exp( - 2.0 * ((double)k) * delta_level * (1.0 - beta_) ) * Ah_level 
		  + std::exp(  2.0 * ((double)k) * delta_level *  beta_ ) * Mh_level  ) );
		  // update with increment
		  GRF_level +=   (direct_solver_.solve( RHS_level0 ));
		  
		  //for -k
		  direct_solver_.compute( ((std::exp(  2.0 * ((double)k) * delta_level * (1.0 - beta_) )) * Ah_level 
		  + (std::exp(  - 2.0 * ((double)k) * delta_level *  beta_ ) ) * Mh_level  ) );
		  // update with increment
		  GRF_level +=   (direct_solver_.solve( RHS_level0 ));
		  
	  }
	  
	  std::clock_t c_end = std::clock();

      double time = 1000000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
      
      cpu_time[0] += time;
 	  
 	  GRF_levels[0] = delta_level * GRF_level;
 	  
 	  
 	 
 	  // loop over levels; actually need BPX PCG here
 	  //std::cout << "start loop comput" << std::endl;
 	  for(int ell=1; ell< numRefines_; ell++)
 	  {
		  std::clock_t c_start = std::clock();
		  
		  //compute RHS
		  vector_t ones((*(Mh_levels[ell])).rows());
		  ones.setOnes();
  
		 //std::cout<<"starting lambda max computation"<<std::endl;
	  
		 lambda_min_max lambda_min_max_(1.0,power_steps);
	  
		 double lambda_max = lambda_min_max_.lambda_max((*(Mh_levels[ell])), ones   );
	  
		 //std::cout<< "lambda max = " << lambda_max <<std::endl;
	  
		 double lambda_min = lambda_min_max_.lambda_min((*(Mh_levels[ell])), ones   );
	  
		 //std::cout<< "lambda min = " << lambda_min <<std::endl;
		 
		 //compute RHS
		 vector_t Y_0((*(Mh_levels[ell])).rows());
		 for(int j=0; j< Y_0.rows(); j++)
		 {
			Y_0(j,0) = dist_norm(generator_norm);
	   
		 } 
		 sqrt_matrix sqrt_matrix_( (lambda_min / 1.5), (lambda_max * 1.5));
		 vector_t RHS_level = sqrt_matrix_.apply_sqrt((10 + ell), (*(Mh_levels[ell])), Y_0);																		
		  
		  
		  //=======================
		  //start sinc quad for k=-K,...,K
		  //=======================
		  
		  vector_t GRF_level( (*(Mh_levels[ell])).cols() );
		  GRF_level.setZero();
		  
		  //int K_level = int(value_of_PI * value_of_PI * std::log( std::sqrt( (RHS_level).rows()  ) )  / ( 4.0 * beta_ )) +1;
		  int K_level = int( ((rate/4.0/(1-beta_)) * std::log( ((double) RHS_level.rows())) ) * ((rate/4.0/(1-beta_)) * std::log( ((double) RHS_level.rows()) )) ) + 1;
		  delta_level = 1.0 / (std::sqrt( ((double)K_level)  ) );
		  
		  
		  
		  // perform sinc for k=1,...,K
		  
		 for(int k=-K_level; k<=K_level; k++)
		 {
			 //std::cout << " k = " << k << std::endl;
			 
			  std::vector< typename std::shared_ptr<matrix_t> > System;
			  // get the right system matrices for BPX
			  for(int i=0; i<=ell;i++)
			  {
				  matrix_t Ah_level = *(Ah_levels[i]);
				  
				  std::shared_ptr<matrix_t> Sh_level_ptr( new matrix_t( 
					(std::exp(-2.0 * ((double) k) * delta_level * (1.0 - beta_) )  * Ah_level  
					+ (std::exp(2.0 * ((double) k) * delta_level * beta_ )) * (*(Mh_levels[i])))  ) );
				  System.push_back( Sh_level_ptr );
				  
			  }
			  
			  // solve with BPX PCG for k == 0
			  solver_t solver_bpx_pcg;
			  auto& preconditioner = solver_bpx_pcg.preconditioner();
				
			  preconditioner.setup( Ih_levels, System, ell+1 );
				
			  solver_bpx_pcg.setMaxIterations( 100 );
			  solver_bpx_pcg.setTolerance( 1.e-12 );
			  
			
			  solver_bpx_pcg.compute( (*(System[ell])) );
		      GRF_level +=  (solver_bpx_pcg.solve( (RHS_level) ));
			 
		 }
		 
		 std::clock_t c_end = std::clock();

         double time = 1000000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
      
         cpu_time[ell] += time;
		 
		 GRF_levels[ell] = delta_level * GRF_level;
	 }
  }
  
  
  for(int i=0; i<cpu_time.size(); i++)
  {
	  cpu_time[i] /= ((double) R);
  }
  
  
  //============================================================================
  // EXPORT GRF
  //============================================================================
  
  // for beta = 0.75
  GRF_levels[numRefines -1] =  (2.0 * std::sin( value_of_PI * beta_ ) / value_of_PI ) * (*(Ih_levels[numRefines -1])) * GRF_levels[numRefines -1];
  
  const vector_t sol_full     = constrained_space.mapToGrid( (GRF_levels[numRefines -1]) );
  
   
  
  
  
  
  
  // interpolate solution
  typedef InterpolationGridFunction< surfaceGridFactory_t,typename surfDH_t::fespace_t,numeric_t>                   func_t;

  
  
  
  func_t      GRF_h      ( surfaceGridFactory, surfDH.fespace(), sol_full );

          
  //============================================================================
  // EXPORTER SECTION 
  //============================================================================
	  
	  
	  if(false)
	  {
  
		  typedef betl2::vtu::Exporter< surfaceGridFactory_t > exporter_t;
		  exporter_t exporter( surfaceGridFactory, basename );
		  exporter
			("GRF"       , GRF_h,       vtu::Entity::Point )
			("kappa"       , kappa,       vtu::Entity::Cell  );
	  }
	  
		
		
	  if(true)
	  {
		  
          
          std::ofstream outFile_dimFE( "results/cpu_time/" + basename_ + "_dimFEspace.txt" );
          // the important part
          for (const auto &e : dimFEspace) outFile_dimFE << e << "\n";
          
          std::ofstream outFile_cpu_time( "results/cpu_time/" + basename_ + "_cpu_time.txt" );
          // the important part
          for (const auto &e : cpu_time) outFile_cpu_time << e << "\n";
          
 	  } 
	 
    
        
  
  //============================================================================
  // A FINAL STATEMENT
  //============================================================================
  
  
  
  
  std::cout << std::endl << "Program terminated normally!" << std::endl;
  
  return 0;
}

