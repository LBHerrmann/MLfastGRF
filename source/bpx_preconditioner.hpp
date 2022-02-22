//! @file   bpx_preconditioner.hpp
//! @author Lukas Herrmann
//! @date   2020

#ifndef BPX_PRECONDITIONER_HPP
#define BPX_PRECONDITIONER_HPP

// Eigen includes --------------------------------------------------------------
#include <Eigen/Dense>

// betl2 includes ---------------------------------------------------------------
#include <functional/grid_function_traits.hpp>

//------------------------------------------------------------------------------


  //----------------------------------------------------------------------------
  template< typename NUMERIC_T >
  class BPXPreconditioner 
  {
  private:
    typedef NUMERIC_T numeric_t;
    typedef Eigen::SparseMatrix<numeric_t> matrix_t;
    typedef Eigen::Matrix< numeric_t, Eigen::Dynamic, 1 > vector_t;
    typedef std::vector< typename std::shared_ptr<matrix_t> > smart_ptr_t;

  private:
    const unsigned int* level_;
    
    const smart_ptr_t* Prolongation_;
    const smart_ptr_t* Ah_;

    

  public:
    /// default constructor
    BPXPreconditioner( )
      : Prolongation_( nullptr  ), Ah_( nullptr ), level_( nullptr )
    { /* empty */ }

    /// forbid copies
    BPXPreconditioner( const BPXPreconditioner& ) = delete;

    /// forbid assignments
    BPXPreconditioner& operator=( const BPXPreconditioner& ) = delete;

    /// destructor
    ~BPXPreconditioner( )
    {
      if( Prolongation_ != nullptr ) delete Prolongation_;
      if( Ah_ != nullptr ) delete Ah_;
      if( level_ != nullptr ) delete level_;
    }

    /// this initializes the preconditioner
    void setup( const smart_ptr_t& Prolongation,
				const smart_ptr_t& Ah,
				const unsigned int& level)
    {
      Prolongation_ = new smart_ptr_t( Prolongation );
      Ah_ = new smart_ptr_t( Ah );
      level_ = new unsigned int(level);
    }

    
    /** @name dummy methods to meet the EIGEN3-requirements */
    //@{
    template< typename MATRIX_T >
    void analyzePattern( const MATRIX_T&/* A */) { /* empty */ }
    template< typename MATRIX_T >
    void factorize     ( const MATRIX_T&/* A */) { /* empty */ }
    template< typename MATRIX_T >
    void compute       ( const MATRIX_T&/* A */) { /* empty */ }
    Eigen::ComputationInfo info() { return Eigen::Success; }
    //@}

    template< typename VECTOR_T >
    vector_t solve( VECTOR_T& b ) const
    {
      
      
      
      vector_t result( (*((*Prolongation_)[0])).cols() );
      result.setZero();
      
      if(*level_ > 1)
      {
		  for(int i=0;i<*level_-1;i++)
		  {
			  //project b onto level 0	  
			  vector_t temp1 = b;
			  for(int k=*level_-2;k>=i;k--)
			  {
				  ETH_ASSERT( ((*((*Prolongation_)[k])).transpose()).cols() == temp1.rows() );
				  temp1 = (*((*Prolongation_)[k])).transpose() * temp1;
				  
			  }
			  
			  vector_t diag = (*((*Ah_)[i])).diagonal();
			  ETH_ASSERT( temp1.rows() == diag.rows() );
			  vector_t temp2( temp1.rows() );
			  for( int j = 0; j < diag.rows(); ++j ) temp2(j) = temp1(j) / diag(j);
			  result += temp2;
			  matrix_t Ih = (*((*Prolongation_)[i]));
			  result = (*((*Prolongation_)[i])) * result;
			  
		  }
	  }
	  
	  vector_t diag = (*((*Ah_)[*level_ - 1])).diagonal();
      ETH_ASSERT( b.rows() == diag.rows() );
      vector_t temp2( b.rows() ); 
	  for( int j = 0; j < diag.rows(); ++j ) temp2(j) = b(j) / diag(j);
	  
	  result += temp2;
	  
      return result;
      //return b;
    }

  }; // end class BPXPreconditioner



#endif // EFIE_BPX_PRECONDITIONER_HPP
