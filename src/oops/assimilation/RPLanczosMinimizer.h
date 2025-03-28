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

#ifndef OOPS_ASSIMILATION_RPLANCZOSMINIMIZER_H_
#define OOPS_ASSIMILATION_RPLANCZOSMINIMIZER_H_

#include <cmath>
#include <string>
#include <vector>

#include "oops/assimilation/CostFunction.h"
#include "oops/assimilation/DualMinimizer.h"
#include "oops/assimilation/DualVector.h"
#include "oops/assimilation/HBHtMatrix.h"
#include "oops/assimilation/MinimizerUtils.h"
#include "oops/assimilation/RinvMatrix.h"
#include "oops/base/IdentityMatrix.h"
#include "oops/util/dot_product.h"
#include "oops/util/Logger.h"
#include "oops/util/printRunStats.h"
#include "oops/util/workflow.h"

namespace oops {

/// RLanczos Minimizer
/*!
 * \brief Augmented Restricted Lanczos method. It is the Lanczos version
 * of RPCG.
 *
 * This solver is based on the algorithm from Gurol, PhD Manuscript,
 * 2013. It performs minimization in the observation space.
 *
 * It solves the linear system \f$ (I + R^{-1}HBH^T) lambda  = H^T R^{-1}d \f$
 * with \f$ HBH^T \f$ inner-product in the augmented observation space.
 *
 * Note that the traditional \f$ B\f$-preconditioning in model space
 * corresponds to \f$I\f$ for this algorithm.
 *
 * A second-level preconditioner, \f$ G \f$, must be symmetric and
 * positive definite with respect to \f$ HBH^T \f$ inner-product.
 * Possible preconditioning is detailed in S. Gurol, PhD Manuscript, 2013.
 *
 * On entry:
 * -    vv      =  starting point, \f$ v_0 \f$.
 * -    vvp     =  starting point augmented part, \f$ vp_0 \f$
 * -    rr      = right hand side.
 * -    aa      = \f$ H (xb - xk) \f$
 * -    sigma   = \f$ (xb - xk)^T B^{-1} (xb - xk)\f$.
 * -    HBHt    = \f$ HBH^T \f$.
 * -    Rinv    = \f$ R^{-1} \f$.
 * -    G       = preconditioner \f$ G \f$.
 *
 * On exit, vv and vvp will contain the solution \f$ lambda = [vv; vp] \f$
 *
 *  The return value is the achieved reduction in preconditioned residual norm.
 *
 *  Iteration will stop if the maximum iteration limit "maxiter" is reached
 *  or if the preconditioned residual norm reduces by a factor of "tolerance".
 */

// -----------------------------------------------------------------------------

template<typename MODEL, typename OBS> class RPLanczosMinimizer : public DualMinimizer<MODEL, OBS> {
  typedef CostFunction<MODEL, OBS>        CostFct_;
  typedef DualVector<MODEL, OBS>          Dual_;
  typedef HBHtMatrix<MODEL, OBS>          HBHt_;
  typedef RinvMatrix<MODEL, OBS>          Rinv_;

 public:
  const std::string classname() const override {return "RPLanczosMinimizer";}
  RPLanczosMinimizer(const eckit::Configuration &, const CostFct_ & J)
    : DualMinimizer<MODEL, OBS>(J) {}
  ~RPLanczosMinimizer() {}

 private:
  double solve(Dual_ &, double &, Dual_ &, const HBHt_ &, const Rinv_ &,
               const double, const double,
               const int &, const double &, Dual_ &, const double &) override;
};

// =============================================================================

template<typename MODEL, typename OBS>
double RPLanczosMinimizer<MODEL, OBS>::solve(Dual_ & vv, double & vvp, Dual_ & rr,
                                        const HBHt_ & HBHt, const Rinv_ & Rinv,
                                        const double costJ0Jb, const double costJ0JoJc,
                                        const int & maxiter, const double & tolerance,
                                        Dual_ & dy, const double & sigma) {
  util::printRunStats("RPLanczos start");

  IdentityMatrix<Dual_> precond;

  Dual_ zz(vv);
  Dual_ ww(vv);
  Dual_ tt(vv);
  Dual_ tt0(vv);
  Dual_ vold(vv);
  Dual_ v(vv);

  // augmented part of the vectors
  double rrp = 1.0;
  double zzp;
  double wwp;
  double ttp = 0.0;
  double ttp0 = 0.0;
  double voldp = 0.0;
  double vp = 1.0;

  std::vector<Dual_> vVEC;  // required for re-orthogonalization
  std::vector<Dual_> tVEC;  // required for re-orthogonalization
  std::vector<Dual_> zVEC;  // required for solution
  // reserve space to avoid extra copies
  vVEC.reserve(maxiter+1);
  tVEC.reserve(maxiter+1);
  zVEC.reserve(maxiter+1);

  std::vector<double> vpVEC;
  std::vector<double> tpVEC;
  std::vector<double> zpVEC;
  std::vector<double> alphas;
  std::vector<double> betas;
  std::vector<double> y;
  std::vector<double> d;

  // J0
  const double costJ0 = costJ0Jb + costJ0JoJc;

  // zzaug = Gaug  rraug
  precond.multiply(rr, zz);
  zzp = rrp;

  // ttaug = HBHtaug zzaug
  HBHt.multiply(zz, tt);
  tt.axpy(zzp, dy);
  ttp = dot_product(dy, zz) + sigma*zzp;

  // Store initial values
  ttp0 = ttp;
  tt0 = tt;

  double normReduction = 1.0;
  double beta0 = sqrt(dot_product(rr, tt) + rrp*ttp);
  double beta = 0.0;

  printNormReduction(0, beta0, normReduction);
  printQuadraticCostFunction(0, costJ0, costJ0Jb, costJ0JoJc);

  vold.zero();
  v = rr;
  vp = rrp;
  v *= 1/beta0;
  vp *= 1/beta0;
  tt *= 1/beta0;
  zz *= 1/beta0;
  ttp *= 1/beta0;
  zzp *= 1/beta0;

  vVEC.clear();
  zVEC.clear();
  tVEC.clear();
  vpVEC.clear();
  zpVEC.clear();
  tpVEC.clear();

  betas.clear();
  alphas.clear();

  vVEC.push_back(v);
  zVEC.push_back(zz);
  tVEC.push_back(tt);
  vpVEC.push_back(vp);
  zpVEC.push_back(zzp);
  tpVEC.push_back(ttp);

  int jiter = 0;
  Log::info() << std::endl;
  while (jiter < maxiter) {
    Log::info() << "RPLanczos Starting Iteration " << jiter+1 << std::endl;

    if (jiter < 5 || (jiter + 1) % 5 == 0) {
      util::update_workflow_meter("iteration", jiter+1);
      util::printRunStats("RPLanczos iteration " + std::to_string(jiter+1));
    }

    // ww = (RinvHBHt + I) zz - beta * vold
    Rinv.multiply(tt, ww);
    ww += zz;
    wwp = zzp;
    ww.axpy(-beta, vold);
    wwp -= beta*voldp;

    double alpha = dot_product(tt, ww) + ttp*wwp;

    ww.axpy(-alpha, v);  // w = w - alpha * v
    wwp -= alpha*vp;

    // Re-orthogonalization
    for (int iiter = 0; iiter < jiter; ++iiter) {
      double proj = dot_product(ww, tVEC[iiter]) + wwp * tpVEC[iiter];
      ww.axpy(-proj, vVEC[iiter]);
      wwp -= proj * vpVEC[iiter];
    }

    // wwaug = Gaug  zzaug
    precond.multiply(ww, zz);
    zzp = wwp;

    // ttaug = HBHtaug zzaug
    HBHt.multiply(zz, tt);
    tt.axpy(zzp, dy);
    ttp = dot_product(dy, zz) + sigma*zzp;

    beta = sqrt(dot_product(tt, ww) + ttp*wwp);

    vold = v;
    voldp = vp;
    v = ww;
    vp = wwp;
    v *= 1/beta;
    vp *= 1/beta;
    tt *= 1/beta;
    zz *= 1/beta;
    ttp *= 1/beta;
    zzp *= 1/beta;

    vVEC.push_back(v);
    zVEC.push_back(zz);
    tVEC.push_back(tt);
    vpVEC.push_back(vp);
    zpVEC.push_back(zzp);
    tpVEC.push_back(ttp);

    alphas.push_back(alpha);

    y.clear();
    if (jiter == 0) {
      y.push_back(beta0/alpha);
    } else {
      // Solve the tridiagonal system T_jiter y_jiter = beta0 * e_1
      d.clear();
      for (int iiter = 0; iiter <= jiter; ++iiter) {
        d.push_back(beta0*(dot_product(tVEC[0], vVEC[iiter]) + tpVEC[0]*vpVEC[iiter]));
      }
      TriDiagSolve(alphas, betas, d, y);
    }

    // The following diagnostic is working only if precond is the IdentityMatrix
    // Compute the quadratic cost function
    // J[du_{i}] = J[0] - 0.5 s_{i}^T Z_{i}^T r_{0}
    // Jb[du_{i}] = 0.5 s_{i}^T V_{i}^T Z_{i} s_{i}
    double costJ = costJ0;

    double costJb = costJ0Jb;
    for (int jj = 0; jj < jiter+1; ++jj) {
      costJ -= 0.5 * y[jj] * dot_product(zVEC[jj], tt0);
      costJ -= 0.5 * y[jj] * zpVEC[jj] * ttp0;
      costJb += 0.5 * y[jj] * dot_product(zVEC[jj], tVEC[jj]) * y[jj];
      costJb += 0.5 * y[jj] * zpVEC[jj] * tpVEC[jj] * y[jj];
    }
    double costJoJc = costJ - costJb;

    // Gradient norm in precond metric --> sqrt(r't) --> beta * y(jiter)
    double rznorm = beta*std::abs(y[jiter]);

    normReduction = rznorm/beta0;

    betas.push_back(beta);

    Log::info() << "RPLanczos end of iteration " << jiter+1 << std::endl;
    printNormReduction(jiter+1, rznorm, normReduction);
    printQuadraticCostFunction(jiter+1, costJ, costJb, costJoJc);

    ++jiter;

    if (normReduction < tolerance) {
      Log::info() << "RPLanczos: Achieved required reduction in residual norm." << std::endl;
      break;
    }
  }

  // Calculate the solution (xh = Binv x)
  for (int iiter = 0; iiter < jiter; ++iiter) {
    vv.axpy(y[iiter], zVEC[iiter]);
    vvp += y[iiter]*zpVEC[iiter];
  }

  Log::info() << "RPLanczos: end" << std::endl;

  util::printRunStats("RPLanczos end");
  return normReduction;
}

// -----------------------------------------------------------------------------

}  // namespace oops

#endif  // OOPS_ASSIMILATION_RPLANCZOSMINIMIZER_H_
