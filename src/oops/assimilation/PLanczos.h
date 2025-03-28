/*
 * (C) Copyright 2009-2016 ECMWF.
 * (C) Crown Copyright 2024, the Met Office.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef OOPS_ASSIMILATION_PLANCZOS_H_
#define OOPS_ASSIMILATION_PLANCZOS_H_

#include <cmath>
#include <vector>

#include "oops/assimilation/MinimizerUtils.h"
#include "oops/assimilation/TriDiagSolve.h"
#include "oops/util/dot_product.h"
#include "oops/util/Logger.h"
#include "oops/util/printRunStats.h"
#include "oops/util/workflow.h"

namespace oops {

/*! \file PLanczos.h
 * \brief Preconditioned Lanczos solver.
 *
 * This algorihtm is based on the standard Preconditioned Lanczos
 * method for solving the linear system Ax = b
 *
 * A must be square and symmetric.
 *
 * On entry:
 * -    x       =  starting point, \f$ x_0 \f$.
 * -    b       = right hand side.
 * -    A       = \f$ A \f$.
 * -    precond = preconditioner \f$ P_k \f$.
 *
 * On exit, x will contain the solution \f$ x \f$
 *
 *  The return value is the achieved reduction in preconditioned residual norm.
 *
 *  Iteration will stop if the maximum iteration limit "maxiter" is reached
 *  or if the residual norm reduces by a factor of "tolerance".
 *
 *  VECTOR must implement:
 *  - dot_product
 *  - operator(=)
 *  - operator(+=),
 *  - operator(-=)
 *  - operator(*=) [double * VECTOR],
 *  - axpy
 *
 *  Each of AMATRIX and PMATRIX must implement a method:
 *  - void multiply(const VECTOR&, VECTOR&) const
 *
 *  which applies the matrix to the first argument, and returns the
 *  matrix-vector product in the second. (Note: the const is optional, but
 *  recommended.)
 */

template <typename VECTOR, typename AMATRIX, typename PMATRIX>
double PLanczos(VECTOR & xx, const VECTOR & bb,
                 const AMATRIX & A, const PMATRIX & precond,
                 const int maxiter, const double tolerance) {
  util::printRunStats("PLanczos start");
  VECTOR zz(xx);
  VECTOR ww(xx);

  std::vector<VECTOR> vVEC;
  std::vector<VECTOR> zVEC;
  // reserve space to avoid extra copies
  vVEC.reserve(maxiter+1);
  zVEC.reserve(maxiter+1);

  std::vector<double> alphas;
  std::vector<double> betas;
  std::vector<double> dd;
  std::vector<double> yy;

  // Initial residual r = b - Ax
  VECTOR rr(bb);
  double xnrm2 = dot_product(xx, xx);
  if (xnrm2 > 0.0) {
    A.multiply(xx, zz);
    rr -= zz;
  }

  // z = precond r
  precond.multiply(rr, zz);

  double normReduction = 1.0;
  double beta0 = sqrt(dot_product(rr, zz));
  double beta = 0.0;

  printNormReduction(0, beta0, normReduction);

  VECTOR vv(rr);
  vv  *= 1/beta0;
  zz  *= 1/beta0;

  zVEC.push_back(zz);     // zVEC[0] = z_1 ---> required for re-orthogonalization
  vVEC.push_back(vv);     // vVEC[0] = v_1 ---> required for re-orthogonalization

  int jiter = 0;
  Log::info() << std::endl;
  while (jiter < maxiter) {
    Log::info() << " PLanczos Starting Iteration " << jiter+1 << std::endl;

    if (jiter < 5 || (jiter + 1) % 5 == 0) {
      util::update_workflow_meter("iteration", jiter+1);
      util::printRunStats("PLanczos iteration " + std::to_string(jiter+1));
    }

    // w = A z - beta * vold
    A.multiply(zz, ww);     // w = A z
    if (jiter > 0) ww.axpy(-beta, vVEC[jiter-1]);

    double alpha = dot_product(zz, ww);

    ww.axpy(-alpha, vv);  // w = w - alpha * v

    // Re-orthogonalization
    for (int iiter = 0; iiter < jiter; ++iiter) {
      double proj = dot_product(ww, zVEC[iiter]);
      ww.axpy(-proj, vVEC[iiter]);
    }

    precond.multiply(ww, zz);  // z = precond w

    beta = sqrt(dot_product(zz, ww));

    vv = ww;
    vv *= 1/beta;
    zz *= 1/beta;

    zVEC.push_back(zz);  // zVEC[jiter+1] = z_jiter
    vVEC.push_back(vv);  // vVEC[jiter+1] = v_jiter

    alphas.push_back(alpha);

    if (jiter == 0) {
      yy.push_back(beta0/alpha);
      dd.push_back(beta0);
    } else {
      // Solve the tridiagonal system T_jiter y_jiter = beta0 * e_1
      dd.push_back(beta0*dot_product(zVEC[0], vv));
      TriDiagSolve(alphas, betas, dd, yy);
    }

    // Gradient norm in precond metric --> sqrt(r'z) --> beta * y(jiter)
    double rznorm = beta*std::abs(yy[jiter]);

    normReduction = rznorm/beta0;

    betas.push_back(beta);

    Log::info() << "PLanczos end of iteration " << jiter+1 << std::endl;
    printNormReduction(jiter+1, rznorm, normReduction);

    ++jiter;

    if (normReduction < tolerance) {
      Log::info() << "PLanczos: Achieved required reduction in residual norm." << std::endl;
      break;
    }
  }

  // Calculate the solution (xh = Binv x)
  for (int iiter = 0; iiter < jiter; ++iiter) {
    xx.axpy(yy[iiter], zVEC[iiter]);
  }

  Log::info() << "PLanczos: end" << std::endl;

  util::printRunStats("PLanczos end");
  return normReduction;
}

}  // namespace oops

#endif  // OOPS_ASSIMILATION_PLANCZOS_H_
