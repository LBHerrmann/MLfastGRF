//! @file   lambda_min_max_SPD.hpp
//! @author Lukas Herrmann
//! @date   2020

#ifndef LAMBDA_MIN_MAX_SPD_HPP
#define LAMBDA_MIN_MAX_SPD_HPP


#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include<chrono>

class lambda_min_max
{
private:
  
  int power_steps_;
  
  typedef double numeric_t;
  typedef Eigen::SparseMatrix<numeric_t> matrix_t;
  typedef Eigen::Matrix<numeric_t,Eigen::Dynamic,1> vector_t;
  
  numeric_t threshold_;

public:
  lambda_min_max( const numeric_t threshold, const int steps ):threshold_( threshold ), power_steps_(steps) {  } 
  

  numeric_t lambda_max(const matrix_t& A, const vector_t& guess)
  {
	  //perform power method first
	  //then perform inverse Reyleigh iteration 
	  
	  
	  vector_t v = guess;
	  
	  
	  
	  //perform power method for fixed number of steps
	  
	  
	  
	  
	  for(int i=0; i<power_steps_; i++)
	  {
		  //update v
		  vector_t w = A * v;
		  double norm = sqrt( w.transpose() * w );
		  v = w /norm;
		  
		  
		    
	  }
	  
	  //approximate eigenvalue
	  
	  
	  numeric_t lambda_max = (v.transpose() * A * v)(0,0) ;
	  
	  //std::cout<< " after "<< power_steps_<< " power iterations lambda max = " << lambda_max <<std::endl;
	  
	  numeric_t lambda_max_temp = 0.0;
	  
	  //improve accuracy by Rayleigh quotient iteration
	  
	  //need CG
	  const int UPLO = Eigen::Lower;
      //typedef Eigen::ConjugateGradient< Eigen::SparseMatrix<numeric_t>, UPLO> solver_t;
      typedef Eigen::MINRES< Eigen::SparseMatrix<numeric_t>> solver_t;
      
      solver_t solver;
      
      solver.setMaxIterations( 20 );
      solver.setTolerance( 1.e-14 );
      
      //define identity matrix
      int n = A.rows(); 
      matrix_t eye(n,n);
      eye.setIdentity();
      
      int i = 0;
	  
	  while (  abs(lambda_max - lambda_max_temp) > threshold_)
	  {
		  
		  lambda_max_temp = lambda_max;
		  
		  matrix_t B = A - lambda_max_temp * eye;
		  solver.compute( B );
		  vector_t w = solver.solveWithGuess(v, v);
		  
		  
		  double norm = sqrt( w.transpose() * w );
		  
		  v = w /norm;
		  
		  
		  
		  lambda_max = (v.transpose() * A * v)(0,0) ;
		  
		  //std::cout << "lambda max = " << lambda_max <<std::endl;
		  
		  i += 1;
		  
		  //std::cout<< "iteration i = " << i <<std::endl;
		  
		  
	  }
	  
	  
	  return lambda_max;
  }
  
  
  numeric_t lambda_min(const matrix_t& A, const vector_t& guess)
  {
	  //perform power method first
	  //then perform inverse Reyleigh iteration 
	  
	  
	  vector_t v = guess;
	  
	  
	  
	  //perform power method for fixed number of steps
	  
	  //need CG
	  const int UPLO = Eigen::Lower;
      //typedef Eigen::ConjugateGradient< Eigen::SparseMatrix<numeric_t>, UPLO> solver_t;
      typedef Eigen::MINRES< Eigen::SparseMatrix<numeric_t>> solver_t;
      solver_t solver;
      
      solver.setMaxIterations( 20 );
      solver.setTolerance( 1.e-14 );
      solver.compute( A );
	  
	  
	  for(int i=0; i<power_steps_; i++)
	  {
		  //update v
		  vector_t w = solver.solveWithGuess(v, v);
		  double norm = sqrt( w.transpose() * w );
		  v = w /norm;
		  
		  
		    
	  }
	  
	  //approximate eigenvalue
	  double norm2 = (v.transpose() * v);
	  
	  numeric_t lambda_min = (v.transpose() * A * v)(0,0) / norm2;
	  
	  numeric_t lambda_min_temp = 0.0;
	  
	  //improve accuracy by Rayleigh quotient iteration
	  
	 
      
      //define identity matrix
      int n = A.rows(); 
      matrix_t eye(n,n);
      eye.setIdentity();
      
      int i = 0;
	  
	  while (  abs(lambda_min - lambda_min_temp) > threshold_)
	  {
		  
		  lambda_min = lambda_min_temp;
		  
		  matrix_t B = A - lambda_min * eye;
		  solver.compute( B );
		  vector_t w = solver.solve(v);
		  
		  double norm = sqrt( w.transpose() * w );
		  
		  v = w /norm;
		  
		  
		  
		  lambda_min_temp = (v.transpose() * A * v)(0,0) ;
		  
		  i += 1;
		  
		  
	  }
	  
	  
	  return lambda_min;
  }
  
  
};

#endif
