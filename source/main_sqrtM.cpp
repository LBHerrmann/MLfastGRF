//! @file   main_sqrtM.hpp
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

// eth includes ----------------------------------------------------------------
#include <input_interface/input_interface.hpp>
#include <grid_utils/grid_view_factory.hpp>
#include <utils/sparsestream.hpp>

// betl2 includes --------------------------------------------------------------
//#include <cmdl_parser/cmdl_parser.hpp>
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





// Own includes
#include "simple_functions.hpp"
#include "lambda_min_max_SPD.hpp"
#include "sqrt_matrix.hpp"


// -----------------------------------------------------------------------------


using namespace betl2;
namespace big = betl2::input::gmsh;


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
        if (-1 == opt){break;}
        switch(opt)
        {
                case 'm': 
                        basename_ = std::string(optarg);
                        //std::cout << "Mesh set to: " << basename_ << std::endl;
                        break;
                case 'n':
                        numRefines_ = std::stoi(optarg);
                        //std::cout << "NumRef set to: " << numRefines_ << std::endl;
                        break;
        }
  }
  
  
  typedef Eigen::SparseMatrix< double > matrix_t;
  typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_t;	
	
  // parse the command line
  const std::string basename = "source/" + basename_;
  //const std::string basename = betl2::parseCommandLine( argc, argv );

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

  


  // define a vector type
  typedef Eigen::Matrix< numeric_t, Eigen::Dynamic, 1 > vector_t;

  //compute mass matrix
  fem::TrivialMaterial<surfaceGridFactory_t>      gamma( surfaceGridFactory, 1.0 );
  massBilinearForm_t M;
  M.compute( constrained_space, gamma );
  M.make_sparse();
  matrix_t Mass = M.matrix();
  
  int dimFEspace = constrained_space.numDofs();
  
  
  
  
  
  
  //random number generator 
  typedef std::mt19937 base_generator_t;
  base_generator_t generator_norm;
  std::normal_distribution<double> dist_norm(0.0, 1.0);
  generator_norm.seed( dimFEspace );
  
  //=========================================
  //test square root of mass matrix here
  //=========================================
  
  //step 1 compute smallest and largest eigenvalue of mass matrix
  
  
  const double eps = std::numeric_limits<double>::epsilon();
  
  const double threshold = 1.0;
  
  int power_steps = 20;
  
  vector_t ones(Mass.rows());
  ones.setOnes();
  
  // approximate smallest and largest eigenvalue of mass matrix
  std::cout<<"starting lambda max computation"<<std::endl;
  
  lambda_min_max lambda_min_max_(threshold,power_steps);
  
  double lambda_max = lambda_min_max_.lambda_max(Mass, ones   );
  
  std::cout<< "lambda max = " << lambda_max <<std::endl;
  
  double lambda_min = lambda_min_max_.lambda_min(Mass, ones   );
  
  std::cout<< "lambda min = " << lambda_min <<std::endl;
  
  // approximate sqrt of mass matrix 
  sqrt_matrix sqrt_matrix_( (lambda_min / 1.1), (lambda_max * 1.1));
  
  //compute directly to compare
  typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_full_t;
  
  vector_t v_test(Mass.rows());
  
  for(int j=0; j< v_test.rows(); j++)
  {
     v_test(j,0) = dist_norm(generator_norm);
   
  }

  vector_t w_ref(Mass.rows());
  
  if(dimFEspace < 2000)
    {matrix_full_t Mass_full = matrix_full_t(Mass);
     w_ref = Mass_full.sqrt() * v_test;}
  else {w_ref = sqrt_matrix_.apply_sqrt(20, Mass, v_test);}

  
  std::vector<double> sqrt_err;
  
  
  for(int i=2; i<10; i++)
  {
  
	  //sqrt_matrix_.test();
	  vector_t w_test = sqrt_matrix_.apply_sqrt(i, Mass, v_test);
	  std::cout<< "succesfully applied sqrt_matrix" << std::endl;
	  
	  
	  
	  vector_t res = w_test - w_ref;
	  
	  sqrt_err.push_back( (res.norm()) / (w_ref.norm())  );
	  
	  std::cout<< " error of sqrtM = " << (res.norm()) / (w_ref.norm())  << " with N = " << i << std::endl;
  }
  
  sqrt_err.push_back( dimFEspace );
  
  
  
  //============================================================================
  // A FINAL STATEMENT
  //============================================================================
  
  
  std::string numRefines_str = std::to_string(numRefines);
  
  std::ofstream outFile_sqrt_err( "results/sqrt_err/sqrt_err_" + numRefines_str + ".txt" );
  // the important part
  for (const auto &e : sqrt_err) outFile_sqrt_err << e << "\n";
         
  
  std::cout << std::endl << "Program terminated normally!" << std::endl;
  
  return 0;
}

