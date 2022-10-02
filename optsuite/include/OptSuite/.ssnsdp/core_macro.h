#ifndef SSNSDP_CORE_MACRO_H
#define SSNSDP_CORE_MACRO_H

#ifndef SSNSDP_DEFAULT_INDEX_TYPE
#define SSNSDP_DEFAULT_INDEX_TYPE ::std::ptrdiff_t
#endif

// Type for scalar value. Typically double.
// Only float and double is supported for MKL and SuiteSparse.
#ifndef SSNSDP_SCALAR_TYPE
#define SSNSDP_SCALAR_TYPE double
#endif

// Index type for sparse matrix. Typically int.
// Only int is supported for SuiteSparse.
#ifndef SSNSDP_SPARSE_INDEX_TYPE
#define SSNSDP_SPARSE_INDEX_TYPE int
#endif

// define to use MKL
// This macro should probably defined by project or cmake,
// instead of defining here.
// #define SSNSDP_USE_MKL

// define to use SuiteSparse
// This macro should probably defined by project or cmake,
// instead of defining here.
// #define SSNSDP_USE_SUITE_SPARSE

// define to disable auto linking for SuiteSparse (msvc only)
// Auto linking should be handled by SuiteSparse... But it isn't.
// #define SSNSDP_DISABLE_SUITE_SPARSE_AUTO_LINK

// define to disable multi threading at compile time
#ifdef SSNSDP_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif

// define to enable the use of std::printf, std::puts
// and anything concerning stdin, stdout, stderr
// if defined to 0, these functions will be implemented
// with C++ style IO
#ifndef SSNSDP_USE_C_STYLE_IO
#define SSNSDP_USE_C_STYLE_IO 1
#endif

// define to use Eigen's built-in eigenvalue solver
// may use MKL's solver if not defined
// You may probably want to define this when using scalar
// type other than float and double.
// #define SSNSDP_FORCE_EIGEN_EIGEN_SOLVER
#if defined(SSNSDP_USE_MKL) && !defined(SSNSDP_FORCE_EIGEN_EIGEN_SOLVER)
#define SSNSDP_MKL_EIG_ENABLED
#endif

// Set to 1 to enable the use of MKL to process sparse matrix
// Poor performance in parallel mode. Not recommended.
#define SSNSDP_USE_MKL_SPARSE 0

#if defined(SSNSDP_USE_MKL) && SSNSDP_USE_MKL_SPARSE
#define SSNSDP_MKL_SPARSE_ENABLED
#endif

#ifndef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE SSNSDP_DEFAULT_INDEX_TYPE
#endif

// define to stop defining NDEBUG when release mode is detected
// #define SSNSDP_NO_DEFINE_NDEBUG

// Compiler detection copied from Eigen
#ifdef __GNUC__
  #define SSNSDP_GNUC 1
#else
  #define SSNSDP_GNUC 0
#endif

#if defined(__clang__)
  #define SSNSDP_CLANG (__clang_major__*100+__clang_minor__)
#else
  #define SSNSDP_CLANG 0
#endif

#if defined(__llvm__)
  #define SSNSDP_LLVM 1
#else
  #define SSNSDP_LLVM 0
#endif

#if defined(__INTEL_COMPILER)
  #define SSNSDP_ICC __INTEL_COMPILER
#else
  #define SSNSDP_ICC 0
#endif

#if defined(__MINGW32__)
  #define SSNSDP_MINGW 1
#else
  #define SSNSDP_MINGW 0
#endif

#if defined(__SUNPRO_CC)
  #define SSNSDP_SUNCC 1
#else
  #define SSNSDP_SUNCC 0
#endif

#if defined(_MSC_VER)
  #define SSNSDP_MSVC _MSC_VER
#else
  #define SSNSDP_MSVC 0
#endif

// Visual Studio 2017 ==> 1910
#if SSNSDP_MSVC && SSNSDP_MSVC < 1910
#error Visual Studio with a version older than 2017 is not supported
#endif

// MSVC does not automatically define NDEBUG in Release configuration
#if SSNSDP_MSVC && !defined(_DEBUG) && !defined(NDEBUG) && !defined(SSNSDP_NO_DEFINE_NDEBUG)
#define NDEBUG
#endif

// define to debug ssnsdp behavior when developing
#if !defined(NDEBUG) && !defined(SSNSDP_INTERNAL_DEBUG)
#define SSNSDP_INTERNAL_DEBUG
#endif

// Global verbosity at compile time
// To change verbosity at runtime, use the verbosity of a model
#ifndef SSNSDP_VERBOSITY
#ifndef NDEBUG
#define SSNSDP_VERBOSITY ::ssnsdp::Verbosity::Debug
#else
#define SSNSDP_VERBOSITY ::ssnsdp::Verbosity::Info
#endif
#endif

// Assertion. Used when wrong input may violate the condition.
// By default, it is simply a wrapper to assert.
// May be overrided to throw an exception, etc.
#ifndef SSNSDP_ASSERT
#define SSNSDP_ASSERT(expr) assert(expr)
#endif

#ifndef SSNSDP_ASSERT_MSG
#define SSNSDP_ASSERT_MSG(expr, msg) assert((expr) && (msg))
#endif

// void expression that marks unreachable code
#if SSNSDP_GNUC || SSNSDP_CLANG
#define SSNSDP_INTERNAL_UNREACHABLE() __builtin_unreachable()
#endif

// Assumption. Used when there is no way to violate the assumption, whatever the input.
// By default, violation of assumptions terminates the program in debug mode,
// and lead to undefined behavior in release mode.
#ifndef SSNSDP_ASSUME
#ifndef NDEBUG
#define SSNSDP_ASSUME(expr) assert(expr)
#else
#if SSNSDP_MSVC
#define SSNSDP_ASSUME(expr) __assume(expr)
#elif defined(SSNSDP_INTERNAL_UNREACHABLE)
#define SSNSDP_ASSUME(expr) ( (expr) ? (void)0 : SSNSDP_INTERNAL_UNREACHABLE() )
#else
#define SSNSDP_ASSUME(expr) ((void)0)
#endif // COMPILER TEST
#endif // #ifndef NDEBUG
#endif // #ifndef SSNSDP_ASSUME

#ifndef SSNSDP_UNREACHABLE
#if defined(SSNSDP_INTERNAL_UNREACHABLE) && defined(NDEBUG)
#define SSNSDP_UNREACHABLE() SSNSDP_INTERNAL_UNREACHABLE()
#else
#define SSNSDP_UNREACHABLE() SSNSDP_ASSUME(false)
#endif
#endif

// encourage compiler to inline a function
#ifndef SSNSDP_STRONG_INLINE
#if SSNSDP_MSVC || SSNSDP_ICC
#define SSNSDP_STRONG_INLINE __forceinline
#else
#define SSNSDP_STRONG_INLINE inline
#endif
#endif

// add an "inline" keyword to specify that multiple definitions are permitted
// Use this macro to state explicitly that inlining is not preferred
#ifndef SSNSDP_WEAK_INLINE
#define SSNSDP_WEAK_INLINE inline
#endif

// Mark a converting constructor as non-explicit
#ifndef SSNSDP_IMPLICIT
#define SSNSDP_IMPLICIT
#endif

// Suppress [[maybe_unused]] has no effect warning of g++
#if SSNSDP_GNUC
#define SSNSDP_CLASS_MEMBER_MAYBE_UNUSED
#else
#define SSNSDP_CLASS_MEMBER_MAYBE_UNUSED [[maybe_unused]]
#endif

#endif
