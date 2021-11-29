cmake_minimum_required (VERSION 2.8)
project(BETL2_MLfastGRF)


set (USE_OPENMP  NO CACHE BOOL "turn on/off OpenMP support" )
set (USE_THREADS NO CACHE BOOL "turn on/off concurrenny" )
set (USE_MKL     NO CACHE BOOL "turn MKL on/off" )
set (USE_UMFPACK NO CACHE BOOL "turn usage of umfpack on/off" )
set (USE_MATIO   NO CACHE BOOL "turn MATLAB IO v1.5.2 support on/off" )

###########################################################################
#
# Set BETL2_SRC_DIR to this directory:
#
###########################################################################
SET(BETL2_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})


###########################################################################
#
# SET DEFAULT BUILD TYPE
#
###########################################################################
if( NOT CMAKE_BUILD_TYPE )
  message(STATUS "No build type selected. Default to DEBUG")
  set (CMAKE_BUILD_TYPE "DEBUG")
endif( )



###########################################################################
#
# DISALLOW IN-SOURCE BUILD
#
###########################################################################
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "NoUnitTests requires an out of source build. Create a seperate binary directory and run CMake there.")
endif("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")


###########################################################################
#
#  SET COMPILER FLAGS WHICH ARE COMMON FOR ALL PROJECTS
#
###########################################################################
include( "${BETL2_SRC_DIR}/../CMake/cxx_flags/CMakeLists.txt" )


###########################################################################
#
#  SET LINEAR ALGEBRA PACKAGE (LINALG_LIBS)
#
###########################################################################
include ("${CMAKE_CURRENT_SOURCE_DIR}/../CMake/Linalg/CMakeLists.linalg")

###########################################################################
#
# SET MODULE PATH
#
###########################################################################
set(CMAKE_MODULE_PATH 
  "${BETL2_SRC_DIR}/../CMake/Modules" ${CMAKE_MODULE_PATH})


###########################################################################
#
# UMFPACK SECTION 
#
###########################################################################
if (USE_UMFPACK)
  find_package (UMFPACK)
  if (UMFPACK_FOUND)
    set (CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -D WITH_UMFPACK")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D WITH_UMFPACK")
    include_directories( ${UMFPACK_INCLUDE_DIR} )
  endif (UMFPACK_FOUND)
endif (USE_UMFPACK)
###########################################################################




INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")
# Tell boost to not automatically link with the library, we do this!
ADD_DEFINITIONS(-D_HAS_CPP0X)

#include boost:
#SET(Boost_DEBUG ON)
#SET(Boost_USE_STATIC_LIBS ON)
FIND_PACKAGE(Boost REQUIRED )
include_directories (${Boost_INCLUDE_DIRS})

# FIND BETL2 LIBS
find_package( BETL2 )
if( BETL2_FOUND )
  INCLUDE_DIRECTORIES( ${BETL2_INCLUDE_DIRS} )
  message( STATUS "BETL2 libs found" )
endif()

# FIND ETH LIBS 
find_package( ETH )
if( ETH_FOUND )
	include_directories( ${ETH_INCLUDE_DIRS} )
  message( STATUS "ETH libs found" )
endif()

# Find Eigen
FIND_PACKAGE(Eigen3 REQUIRED)
IF(EIGEN3_FOUND)
	INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR})
  SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN3_ACTIVATED" )
  message( STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}" )
ENDIF()


# Find algo644
FIND_PACKAGE( ALGO644 )

if( UNIX AND NOT APPLE )
set( GFORTRAN_LIB gfortran )
endif (UNIX AND NOT APPLE )

if( APPLE )
find_library( GFORTRAN_LIB NAMES gfortran 
  libgfortran.a HINTS "/opt/local/lib/gcc48" "/usr/local/lib/gcc/4.9" "/usr/local/lib/gcc/5" "/usr/lib64" "$ENV{GFORTRAN_ROOT}")
endif( APPLE )

# Concatenate libs
set( ALL_LIBS 
  ${BETL2_LIBRARIES} 
  ${ETH_LIBRARIES} 
  ${Boost_LIBRARIES} 
  ${LINALG_LIBRARIES} 
  ${ALGO644_LIBRARIES} 
  ${GFORTRAN_LIB}) # ${CMAKE_THREAD_LIBS_INIT} )

if (MATIO_FOUND AND ZLIB_FOUND AND HDF5_FOUND )
  set (ALL_LIBS ${ALL_LIBS} ${MATIO_LIBRARIES} ${ZLIB_LIBRARIES} ${HDF5_LIBRARIES}) 
endif (MATIO_FOUND AND ZLIB_FOUND AND HDF5_FOUND )


if( MKL_FOUND )
  set( ALL_LIBS ${ALL_LIBS} ${MKL_LIBRARIES} )
endif( MKL_FOUND )

if( UMFPACK_FOUND )
  set( ALL_LIBS ${ALL_LIBS} ${UMFPACK_LIB} )
endif( UMFPACK_FOUND )

if (MATIO_FOUND AND ZLIB_FOUND AND HDF5_FOUND )
  set (ALL_LIBS ${ALL_LIBS} ${MATIO_LIBRARIES} ${ZLIB_LIBRARIES} ${HDF5_LIBRARIES}) 
endif (MATIO_FOUND AND ZLIB_FOUND AND HDF5_FOUND )

if( Threads_FOUND AND METIS_FOUND )
  set( ALL_LIBS ${CMAKE_THREAD_LIBS_INIT} ${METIS_LIBRARIES} ${ALL_LIBS})
endif( Threads_FOUND AND METIS_FOUND )



############################################################################
#
# Add executable specific information
#
#############################################################################

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    add_definitions(-Wno-deprecated-declarations -Wno-ignored-attributes -Wno-misleading-indentation -Wno-unknown-pragmas)
endif()

# Boost
find_package(Boost REQUIRED COMPONENTS program_options serialization mpi)
# include dirs
include_directories(${Boost_INCLUDE_DIRS})
# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})


include_directories(${BETL2_SRC_DIR}/genericMLMC_v1.1.0/include)

# define the executable name
set (EXECNAME_SQRT betl2_fastGRF_err_sqrtM)
set (EXECNAME_L2L2_POLYGON betl2_fastGRF_L2L2_err_polygon)
set (EXECNAME_CPU_TIME betl2_fastGRF_cpu_time)

# create betl executable
add_executable (${EXECNAME_SQRT}
  ${BETL2_SRC_DIR}/source/main_sqrtM.cpp
  )
  
target_link_libraries ( ${EXECNAME_SQRT} 
			${ALL_LIBS} 
			${MPI_LIBRARIES}
			${Boost_LIBRARIES}
				)

set_target_properties (${EXECNAME_SQRT} 
 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BETL2_SRC_DIR}/source)
				
add_executable (${EXECNAME_L2L2_POLYGON}
  ${BETL2_SRC_DIR}/source/main_fastGRF_L2L2_error.cpp
  ) 

target_link_libraries ( ${EXECNAME_L2L2_POLYGON} 
			${ALL_LIBS} 
			${MPI_LIBRARIES}
			${Boost_LIBRARIES}
				)

set_target_properties (${EXECNAME_L2L2_POLYGON} 
 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BETL2_SRC_DIR}/source)
 
add_executable (${EXECNAME_CPU_TIME}
  ${BETL2_SRC_DIR}/source/main_fastGRF_cpu_time.cpp
  ) 

target_link_libraries ( ${EXECNAME_CPU_TIME} 
			${ALL_LIBS} 
			${MPI_LIBRARIES}
			${Boost_LIBRARIES}
				)

#set_target_properties (${EXECNAME_CPU_TIME} 
# PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_PATH}) 
set_target_properties (${EXECNAME_CPU_TIME}
PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BETL2_SRC_DIR}/source)
