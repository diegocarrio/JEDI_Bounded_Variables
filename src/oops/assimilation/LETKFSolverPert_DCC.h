/*
 * (C) Copyright 2020 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */


#ifndef OOPS_ASSIMILATION_LETKFSOLVERPERT_H_
#define OOPS_ASSIMILATION_LETKFSOLVERPERT_H_

#include <Eigen/Dense>
#include <cfloat>
#include <memory>
#include <string>
#include <vector>

#include "eckit/config/Configuration.h"
#include "oops/assimilation/LETKFSolver.h"
#include "oops/base/Departures.h"
#include "oops/base/DeparturesEnsemble.h"
#include "oops/base/Geometry.h"
#include "oops/base/IncrementEnsemble4D.h"
#include "oops/base/ObsErrors.h"
#include "oops/base/Observations.h"
#include "oops/base/ObsLocalizations.h"
#include "oops/base/ObsSpaces.h"
#include "oops/interface/GeometryIterator.h"
#include "oops/util/Logger.h"

namespace oops {
  class Variables;

/*!
 * An implementation of the perturbed form of the LETKF from Hunt et al. 2007
 */


template <typename MODEL, typename OBS>
class LETKFSolverPert : public LETKFSolver<MODEL, OBS> {
  typedef Departures<OBS>              Departures_;
  typedef DeparturesEnsemble<OBS>      DeparturesEnsemble_;
  typedef Geometry<MODEL>              Geometry_;
  typedef GeometryIterator<MODEL>      GeometryIterator_;
  typedef ObsErrors<OBS>               ObsErrors_;
  typedef ObsSpaces<OBS>               ObsSpaces_;
  typedef State4D<MODEL>               State4D_;
  typedef StateEnsemble4D<MODEL>       StateEnsemble4D_;
  typedef IncrementEnsemble4D<MODEL>   IncrementEnsemble4D_;
  typedef ObsLocalizations<MODEL, OBS> ObsLocalizations_;
  typedef Observations<OBS>            Observations_;

 public:
  static const std::string classname() {return "oops::LETKFSolverPert";}

  LETKFSolverPert(ObsSpaces_ &, const Geometry_ &, const eckit::Configuration &, size_t,
              const State4D_ &, const Variables &);


  Observations_ computeHofX(const StateEnsemble4D_ &, size_t, bool) override;


  /// KF update + posterior inflation at a grid point location (GeometryIterator_)
  //void measurementUpdate(const IncrementEnsemble4D_ &,
  //                       const GeometryIterator_ &, IncrementEnsemble4D_ &) override;

  //private:
  /// Computes weights for ensemble update with local observations
  /// \param[in] omb      Observation departures (nlocalobs)
  /// \param[in] Yb       Ensemble perturbations (nens, nlocalobs)
  /// \param[in] invvarR  Inverse of observation error variances (nlocalobs)
  //virtual void computeWeights(const Eigen::VectorXd & omb, const Eigen::MatrixXd & Yb,
  //                            const Eigen::VectorXd & invvarR);

  /// Applies weights and adds posterior inflation
  //virtual void applyWeights(const IncrementEnsemble4D_ &, IncrementEnsemble4D_ &,
  //                          const GeometryIterator_ &);

  private:
  // eigen solver matrices
  Eigen::VectorXd eival_;
  Eigen::MatrixXd eivec_;

  DeparturesEnsemble_ YobsPert_;
  eckit::LocalConfiguration observationsConfig_;


};


// -----------------------------------------------------------------------------
template <typename MODEL, typename OBS>
LETKFSolverPert<MODEL, OBS>::LETKFSolverPert(ObsSpaces_ & obspaces, const Geometry_ & geometry,
                                     const eckit::Configuration & config, size_t nens,
                                     const State4D_ & xbmean, const Variables & incvars)
  : LETKFSolver<MODEL, OBS>(obspaces, geometry, config, nens, xbmean, incvars),
    eival_(this->nens_), eivec_(this->nens_, this->nens_),
    YobsPert_(obspaces, this->nens_),
    observationsConfig_(config.getSubConfiguration("observations"))
{
  Log::trace() << "LETKFSolverPert<MODEL, OBS>::create starting" << std::endl;
  Log::info() << "Using EIGEN implementation of the PERTURBED LETKF" << std::endl;
  Log::trace() << "LETKFSolverPert<MODEL, OBS>::create done" << std::endl;
}


// -----------------------------------------------------------------------------
template <typename MODEL, typename OBS>
Observations<OBS> LETKFSolverPert<MODEL, OBS>::computeHofX(const StateEnsemble4D_ & ens_xx,
		                                          size_t iteration, bool readFromFile) {
        util::Timer timer(classname(), "computeHofX");
	Observations_ yb_mean(this->obspaces_);
        yb_mean = LETKFSolver<MODEL, OBS>::computeHofX(ens_xx, iteration, readFromFile);
	Log::info() << "DCC: yb_mean: " << yb_mean << std::endl;
//	Log::info() << "DCC: iteration: " << iteration << std::endl;
	Log::info() << "DCC: this->nens_ : " << this->nens_ << std::endl;


	// Recover the full HofX from the original ensemble and store them in (this->Yb_):
	for (size_t iens = 0; iens < (this->nens_); ++iens) {
	    (this->Yb_)[iens] += yb_mean.obstodep();
	}

	
	// Store original observation in (this->omb):
	Observations_ yobs(this->obspaces_, "ObsValue")
	(this->omb) = yobs.obstodep();



        // Loop through ensemble to generate perturbed observations:
	//
	Departures_ yobsPertSum(this->obspaces_); // Create object yobsPertSum that will contain the sum of the perturbed observations
	for (size_t iens = 0; iens < (this->nens_); ++iens) {
	    Observations_ ypert(yobs);
	    ypert.perturb(*(this->R_));
	    YobsPert_[iens] = ypert.obstodep();
	    YobsPert_[iens].mask(this->Yb_[iens]);
	    yobsPertSum += YobsPert_[iens];
	}
	//
	// Recenter the perturbed observations around original obs:
	//
	for (size_t iens = 0; iens < (this->nens_); ++iens) {
	    YobsPert_[iens].axpy(-1.0/(this->nens_), yobsPertSum);
            YobsPert_[iens] += yobs.obstodep();
	}


        return yb_mean;
}







// -----------------------------------------------------------------------------
//template <typename MODEL, typename OBS>
//void LETKFSolverPert<MODEL, OBS>::measurementUpdate(const IncrementEnsemble4D_ & bkg_pert,
//                                                    const GeometryIterator_ & i,
//                                                    IncrementEnsemble4D_ & ana_pert) {
//        util::Timer timer(classname(), "measurementUpdate");
//        Log::info() << "DCC: Using Measurement Update Function" << std::endl;
//
//}



}  // namespace oops
#endif  // OOPS_ASSIMILATION_LETKFSOLVER_H_
