//! @file   sqrt_matrix.hpp
//! @author Lukas Herrmann
//! @date   2020

#ifndef SQRT_MATRIX_HPP
#define SQRT_MATRIX_HPP


#include <Eigen/IterativeLinearSolvers>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/jacobi_elliptic.hpp>

#include<chrono>

class sqrt_matrix
{
private:
  
  
  
  
  
  
  typedef double numeric_t;
  
  
  numeric_t lambda_min_;
  numeric_t lambda_max_;
  typedef Eigen::SparseMatrix<numeric_t> matrix_t;
  
  
  
  
  
  typedef Eigen::Matrix<numeric_t,Eigen::Dynamic,1> vector_t;
  
  const numeric_t value_of_PI = 3.141592653589793238462643383279;
  

public:
  sqrt_matrix( numeric_t lambda_min, numeric_t lambda_max):
  lambda_min_(lambda_min), lambda_max_(lambda_max){  } 
  

  void test() 
  {
	  
	  std::cout << "ellipk test: " << boost::math::ellint_1(std::sqrt( 0.5 ), value_of_PI * 0.5 ) << std::endl;
	  std::cout << "sn = " << boost::math::jacobi_sn(0.5, std::sqrt(0.5) ) << std::endl;
	  std::cout << "cn = " << boost::math::jacobi_cn(0.5, std::sqrt(0.5) ) << std::endl;
	  std::cout << "dn = " << boost::math::jacobi_dn(0.5, std::sqrt(0.5) ) << std::endl;
  }
  
  vector_t apply_sqrt(const int N_, const matrix_t& A, const vector_t& v)
  {
	  
	  //need CG
	  const int UPLO = Eigen::Lower;
      typedef Eigen::ConjugateGradient< Eigen::SparseMatrix<numeric_t>, UPLO> solver_t;
      solver_t solver;
      
      matrix_t eye(A.rows(),A.cols());
      eye.setIdentity();
	  
	  vector_t res(v.rows());
	  res.setZero();
	  
	  numeric_t k2 = lambda_min_ / lambda_max_;
	  numeric_t Kp = boost::math::ellint_1(std::sqrt( 1 - k2 ), value_of_PI * 0.5 );
	  
	  //test
	  //std::cout << "Kp = " << Kp << std::endl;
	  
	  for(int j=1;j<=N_; j++)
	  {
		  numeric_t tj = (((double) j) - 0.5) * Kp /((double) N_);
		  numeric_t sn = boost::math::jacobi_sn(std::sqrt(1.0 - k2), tj );
		  numeric_t cn = boost::math::jacobi_cn(std::sqrt(1.0 - k2), tj );
		  numeric_t dn = boost::math::jacobi_dn(std::sqrt(1.0 - k2), tj );
		  
		  
		  cn = (1.0/cn);
		  dn *= cn;
		  // dont write imaginary unit, put a minus sign later
		  sn *= cn;
		  numeric_t w = std::sqrt( lambda_min_ ) * sn;
		  numeric_t dzdt = cn * dn;
		  
		  solver.compute( A + (w*w) * eye);
		  
		  res -= solver.solve(v) * dzdt;
		  
	  }
	  
	  res = (-2.0*Kp*std::sqrt(lambda_min_)) / (value_of_PI* ((double)N_) ) * (A * res);
	  
	  
	  return res;
  }
  
  
  
};

#endif
