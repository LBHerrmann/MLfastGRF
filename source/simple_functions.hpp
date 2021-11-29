//! @file   simple_functions.hpp
//! @author Lukas Herrmann
//! @date   2020

#ifndef SIMPLE_FUNCTIONS_HPP
#define SIMPLE_FUNCTIONS_HPP

// system includes -------------------------------------------------------------
#include <cmath>
#include <random>

// Eigen includes --------------------------------------------------------------
#include <Eigen/Dense>

// betl2 includes --------------------------------------------------------------
#include <functional/grid_function.hpp>
#include <functional/grid_function_traits.hpp>


//------------------------------------------------------------------------------
namespace betl2 {
  namespace fem {
    
    template< typename GRID_VIEW_FACTORY_T >
    class ReactionTerm
      : public GridFunction< GridFunctionTraits< typename GRID_VIEW_FACTORY_T::gridTraits_t,
                                                 double,
                                                 1 >,               //function dimension: it's a scalar, so 1
                             ReactionTerm<GRID_VIEW_FACTORY_T> 
                             >
    {
    public:
      typedef GridFunctionTraits< typename GRID_VIEW_FACTORY_T::gridTraits_t,
                                  double,1  > gridFunctionTraits_t; //idem here

    private:
      typedef GridFunction< gridFunctionTraits_t, ReactionTerm<GRID_VIEW_FACTORY_T> > base_t;
      typedef typename gridFunctionTraits_t::entityCollection_t entityCollection_t;

    public:
      /** @name static data and typedefs needed to meet GridFunction requirements */
      //@{
      /// Type for local coordinates \f$ x \f$, input for function evaluation
      typedef typename gridFunctionTraits_t::first_argument_t  first_argument_t;
      /// Type for element representation
      typedef typename gridFunctionTraits_t::second_argument_t second_argument_t;
      /// Type of the result of the evaluation \f$ f(x) \f$ (a vector of length \ref dimTo).
      typedef typename gridFunctionTraits_t::result_t          result_t;  
      /// Allows for some optimization in the evaluation of bilinear forms
      static constexpr bool isScalar( )            { return true; }
      /// Allows for some optimization in the evaluation of bilinear forms
      static constexpr bool isPiecewiseConstant( ) { return true; }
      //@}

    private:
      static const int dimTo = gridFunctionTraits_t::dimTo;
      const entityCollection_t entity_collection_;
      result_t                 kappa1_;
      result_t                 kappa2_;
      
      

    public:
      ReactionTerm( const GRID_VIEW_FACTORY_T& gridViewFactory, double kappa1 = 1.0, double kappa2 = 1.0 )
        : base_t           ( )
        , entity_collection_( gridViewFactory.getView().template entities<0>() )
        , kappa1_( )
        , kappa2_( )
      {
        kappa1_ << kappa1;
        kappa2_ << kappa2;
      }

      inline result_t operator()( first_argument_t  gp , second_argument_t  E  ) const
      {
	    first_argument_t globalCoord = E.geometry().global( gp );
	    
	    
	    if( globalCoord(0)  < 0.5)
	    {
			return kappa1_;
		}
		else
		{
			return kappa2_;
		}
		  
        
      }

      /** @name iterator support */
      //@{
      typename gridFunctionTraits_t::elementIterator_t begin() const { return entity_collection_.begin(); }
      typename gridFunctionTraits_t::elementIterator_t end()   const { return entity_collection_.end();   }
      //@}
    }; // end class ReactionTerm


  }
}

#endif
