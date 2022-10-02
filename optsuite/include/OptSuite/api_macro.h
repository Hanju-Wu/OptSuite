/*
 * ==========================================================================
 *
 *       Filename:  api_macro.h
 *
 *    Description:  header defines API macros for OptSuite
 *
 *        Version:  1.0
 *        Created:  10/07/2021 04:15:50 PM
 *       Revision:  none
 *       Compiler:  gcc/g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifndef OPTSUITE_API_MACRO_H
#define OPTSUITE_API_MACRO_H

// for windows 
#if defined(__WIN32__) || defined(__CYGWIN__)
  #define OPTSUITE_HELPER_DLL_IMPORT __declspec(dllimport)
  #define OPTSUITE_HELPER_DLL_EXPORT __declspec(dllexport)
  #define OPTSUITE_HELPER_DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define OPTSUITE_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
    #define OPTSUITE_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
    #define OPTSUITE_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define OPTSUITE_HELPER_DLL_IMPORT
    #define OPTSUITE_HELPER_DLL_EXPORT
    #define OPTSUITE_HELPER_DLL_LOCAL
  #endif
#endif

#ifdef OPTSUITE_BUILD_DYN_LIB
  #if defined(OPTSUITE_DLL_EXPORTS) || defined(OptSuite_EXPORTS)
    #define OPTSUITE_API OPTSUITE_HELPER_DLL_EXPORT
  #else
    #define OPTSUITE_API OPTSUITE_HELPER_DLL_IMPORT
  #endif
  #define OPTSUITE_LOCAL OPTSUITE_HELPER_DLL_LOCAL
#else
  #define OPTSUITE_API
  #define OPTSUITE_LOCAL
#endif

#endif

