//! @file   compute_fastGRF_cpu_time.cpp
//! @author Lukas Herrmann
//! @date   2020

typedef Eigen::SparseMatrix< double > matrix_t;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_t;

void compute_fastGRF_cpu_time(double beta,
                              int rep,
                              int numRefines,
                              const std::vector< typename std::shared_ptr<matrix_t> > &Ih_levels,
                              const std::vector< typename std::shared_ptr<matrix_t> > &Ah_levels,
                              const std::vector< typename std::shared_ptr<matrix_t> > &Mh_levels,
                              std::vector<double> &cpu_time
                             )
{
  typedef double numeric_t;
  
  //========================================
  // test sqrt M algorithm
  //======================================== 

  //random number generator for the seed
  typedef std::mt19937 base_generator_t;
  base_generator_t generator_norm;
  std::normal_distribution<double> dist_norm(0.0, 1.0);
  generator_norm.seed( 5489 );
  
  
  
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
  
  
  
  int R = rep;
  
  const numeric_t value_of_PI = 3.141592653589793238462643383279;
  numeric_t beta_ = beta;
  
  numeric_t rate = 2.0 * beta_ - 1.0;
  
  const int UPLO = Eigen::Lower;
  typedef BPXPreconditioner< numeric_t > precond_t;
  typedef Eigen::ConjugateGradient< Eigen::SparseMatrix<numeric_t>, UPLO, precond_t > solver_t;
  
  std::vector< vector_t > GRF_levels( numRefines );
  
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
 	  for(int ell=1; ell< numRefines_; ell++)
 	  {
		  std::clock_t c_start = std::clock();
		  
		  //compute RHS
		  vector_t ones((*(Mh_levels[ell])).rows());
		  ones.setOnes();
	  
		  lambda_min_max lambda_min_max_(1.0,power_steps);
	  
		  double lambda_max = lambda_min_max_.lambda_max((*(Mh_levels[ell])), ones   );
	  
		  double lambda_min = lambda_min_max_.lambda_min((*(Mh_levels[ell])), ones   );
		 
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
  
}
