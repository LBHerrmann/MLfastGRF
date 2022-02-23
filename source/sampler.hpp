//! @file   sampler.hpp
//! @author Lukas Herrmann
//! @date   2020

#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include<chrono>
#include <random>

class samplerclass
{
private:
    unsigned K_;
    unsigned seed_;

    typedef std::mt19937 base_generator_t;
    base_generator_t generator;

public:
    samplerclass( const unsigned K, const unsigned seed ):K_( K ), seed_( seed )
    {
        generator.seed(seed_ /*+ std::chrono::system_clock::now().time_since_epoch().count()*/ );
    }


    std::vector< double > operator()(int i )
    {



        std::normal_distribution<double> dist(0.0, 1.0);



        std::vector< double > y( K_ );
        for (int kk=0; kk<K_; kk++) {
            //y[kk] = 0.5;
            y[kk] = dist(generator);
        }

        return y;
    }
};

#endif
