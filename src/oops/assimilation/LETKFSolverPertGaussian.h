/*
 * (C) Copyright 2020 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */


#ifndef OOPS_ASSIMILATION_LETKFSOLVERPERTGAUSSIAN_H_
#define OOPS_ASSIMILATION_LETKFSOLVERPERTGAUSSIAN_H_

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
 * An implementation of the Gaussian perturbed form of the LETKF from Hunt et al. 2007
 */

template <typename MODEL, typename OBS>
class LETKFSolverPertGaussian : public LETKFSolver<MODEL, OBS> {
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
  static const std::string classname() {return "oops::LETKFSolverPertGaussian";}

  LETKFSolverPertGaussian(ObsSpaces_ &, const Geometry_ &, const eckit::Configuration &, size_t,
              const State4D_ &, const Variables &);


  Observations_ computeHofX(const StateEnsemble4D_ &, size_t, bool) override;


  /// KF update + posterior inflation at a grid point location (GeometryIterator_)
  void measurementUpdate(const IncrementEnsemble4D_ &,
                         const GeometryIterator_ &, IncrementEnsemble4D_ &) override;

  //private:
  /// Computes weights for ensemble update with local observations
  /// \param[in] omb      Observation departures (nlocalobs)
  /// \param[in] Yb       Ensemble perturbations (nens, nlocalobs)
  /// \param[in] invvarR  Inverse of observation error variances (nlocalobs)
  virtual void computeWeights(const Eigen::MatrixXd & YobsPert, const Eigen::MatrixXd & Yb,
                              const Eigen::MatrixXd & HXb,  const Eigen::VectorXd & diagInvR);


  /// Applies weights and adds posterior inflation
  virtual void applyWeights(const IncrementEnsemble4D_ &, IncrementEnsemble4D_ &,
                            const GeometryIterator_ &);


  // Protected variables:
  protected:
  DeparturesEnsemble_ HXb_;    ///< full background ensemble in the observation space HXb;


  // Private variables:
  private:
  DeparturesEnsemble_ YobsPertDepEns_;

  eckit::LocalConfiguration observationsConfig_; // Configuration for observations


};


// -----------------------------------------------------------------------------
//  Constructor of the LETKF Pert Gaussian solver:
template <typename MODEL, typename OBS>
LETKFSolverPertGaussian<MODEL, OBS>::LETKFSolverPertGaussian(ObsSpaces_ & obspaces, const Geometry_ & geometry,
                                     const eckit::Configuration & config, size_t nens,
                                     const State4D_ & xbmean, const Variables & incvars)
  : LETKFSolver<MODEL, OBS>(obspaces, geometry, config, nens, xbmean, incvars),
    HXb_(obspaces, this->nens_),
    YobsPertDepEns_(obspaces, this->nens_),
    observationsConfig_(config.getSubConfiguration("observations"))
{
  Log::trace() << "LETKFSolverPertGaussian<MODEL, OBS>::create starting" << std::endl;
  Log::info()  << "Using EIGEN implementation of the PERTURBED LETKF"    << std::endl;
  Log::trace() << "LETKFSolverPertGaussian<MODEL, OBS>::create done"     << std::endl;
}


// -----------------------------------------------------------------------------
//  Compute HofX for the perturbed LETKF:
template <typename MODEL, typename OBS>
Observations<OBS> LETKFSolverPertGaussian<MODEL, OBS>::computeHofX(const StateEnsemble4D_ & ens_xx,
		                                          size_t iteration, bool readFromFile) {
        util::Timer timer(classname(), "computeHofX");

	Observations_ yb_mean(this->obspaces_);
        yb_mean = LETKFSolver<MODEL, OBS>::computeHofX(ens_xx, iteration, readFromFile);
	Log::info() << "----------------------------------" << std::endl;	
	Log::info() << "DCC: COMPUTING HofX" << std::endl;

        // Recover the full HofX from the original ensemble and store them in (HXb_):
	// Remember: Yb_ stands for ensemble perturbations in the observation space: Yb_ = HXb_ - Hxb
	// Thus, HXb = Hxb + Yb_
        for (size_t iens = 0; iens < (this->nens_); ++iens) {
	    HXb_[iens] = yb_mean + (this->Yb_)[iens];   
        }

	// Store original observation in (this->omb_):
        Observations_ yobs(this->obspaces_, "ObsValue");
	(this->omb_) = yobs.obstodep();


        // Loop through ensemble to generate perturbed observations:
        //
	Observations_ ypertObsTmp(this->obspaces_, "ObsValue");
	Observations_ yzeroObsTmp(ypertObsTmp);
	yzeroObsTmp.zero();

	Departures_ ypertDepSum(this->obspaces_); // Create object yobsPertSum that will contain the sum of the perturbed observations
        ypertDepSum.zero();
	for (size_t iens = 0; iens < (this->nens_); ++iens) {	    	
	    ypertObsTmp.perturb(*(this->R_));
	    YobsPertDepEns_[iens] = ypertObsTmp - yzeroObsTmp;
            ypertDepSum += YobsPertDepEns_[iens];       // ypertDepSum contains the sum of the perturbed observations
        }	


	//
        // Recenter the perturbed observations around original obs:
        //
        for (size_t iens = 0; iens < (this->nens_); ++iens) {
            YobsPertDepEns_[iens].axpy(-1.0/(this->nens_), ypertDepSum);
            YobsPertDepEns_[iens] += yobs.obstodep();
        }

        return yb_mean;
}


// -----------------------------------------------------------------------------
//  Measurement update for the perturbed LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverPertGaussian<MODEL, OBS>::measurementUpdate(const IncrementEnsemble4D_ & bkg_pert,
		                                    	    const GeometryIterator_ & i,
						    	    IncrementEnsemble4D_ & ana_pert) {

      util::Timer timer(classname(), "measurementUpdate");
      Log::info() << "DCC: Using measurementUpdate in the PERTURBED LETKF GAUSSIAN" << std::endl;
      
      // Create the local subset of observations:
      Departures_ locvector(this->obspaces_);
      locvector.ones();
      this->obsloc().computeLocalization(i, locvector);

      // Set invVarR_ to missing values wherever they are missing in HXb_[iens] to avoid weigth computation errors:
      for (size_t iens = 0; iens < (this->nens_); ++iens) {
	   (this->invVarR_)->mask(this->HXb_[iens]);
      }

      locvector.mask(*(this->invVarR_));      
      const Eigen::VectorXd local_omb_vec = this->omb_.packEigen(locvector);

      if (local_omb_vec.size() == 0) {
	   Log::info() << "DCC: No observations found in this local volume. No need to update Wa_ and wa_" << std::endl;
	   this->copyLocalIncrement(bkg_pert, i, ana_pert);
      } else { 	      
	      Log::info() << "DCC: Obs found in this local volume. Do normal LETKF update" << std::endl;
	      Eigen::MatrixXd local_Yb_mat = (this->Yb_).packEigen(locvector); // the Eigen::MatrixXd function is used to convert the DeparturesEnsemble_ to Eigen::MatrixXd
	      Eigen::MatrixXd local_YobsPert_mat = YobsPertDepEns_.packEigen(locvector);
	      Eigen::MatrixXd local_HXb_mat = (this->HXb_).packEigen(locvector);
	      // Create local obs errors:
	      Eigen::VectorXd local_invVarR_vec = (this->invVarR_)->packEigen(locvector); 
	      // Apply localization:
	      Eigen::VectorXd localization = locvector.packEigen(locvector);
	      local_invVarR_vec.array() *= localization.array();
              computeWeights(local_YobsPert_mat, local_Yb_mat, local_HXb_mat, local_invVarR_vec);
	      applyWeights(bkg_pert, ana_pert, i);
      }
}


// -----------------------------------------------------------------------------
// Compute weights for the perturbed LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverPertGaussian<MODEL, OBS>::computeWeights(const Eigen::MatrixXd & YobsPert,
						 	 const Eigen::MatrixXd & Yb,
						 	 const Eigen::MatrixXd & HXb,
						 	 const Eigen::VectorXd & InvVarR){

	util::Timer timer(classname(), "computeWeights");
        Log::info() << "DCC: Using computeWeights in the PERTURBED LETKF GAUSSIAN" << std::endl;

	// Compute transformation matix, save in Wa_, wa_
	// Uses C++ eigen interface
	// Implements perturbed observation version of LETKF from Hunt et al. 2007
	const LocalEnsembleSolverInflationParameters & inflopt = (this->options_).infl;
        const double infl = inflopt.mult;

	// Compute work = (Yb^T) R^(-1) Yb + [(nens-1)/infl I]
	Eigen::MatrixXd work = Yb*(InvVarR.asDiagonal()*Yb.transpose());
	work.diagonal() += Eigen::VectorXd::Constant(this->nens_, (this->nens_-1)/infl);

	// Compute eigenvalues and eigenvectors of the above matrix:
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(work);
	this->eival_ = es.eigenvalues().real();
	this->eivec_ = es.eigenvectors().real();

	// Computing Pa_tilde = [ (Yb^T) R^(-1) Yb + (nens-1)/infl I  ]^(-1):
	work.noalias() = (this->eivec_) * (this->eival_).cwiseInverse().asDiagonal() * (this->eivec_).transpose();    // cwiseInverse() computes the inverse of the eigenvalues 

	// Computing matrix form of wa_tilde = Pa_tilde * (Yb^T) * R^(-1) * (YobsPert - HXb):
	// We store this object into (this->Wa_):
	(this->Wa_).noalias() = work * ( Yb * InvVarR.asDiagonal() * (YobsPert - HXb).transpose() );

	Log::info() << "DCC: ComputeWeights in the PERTURBED LETKF GAUSSIAN COMPLETED" << std::endl;

} // End function computeWeights


// -----------------------------------------------------------------------------
// Apply weights and add posterior inflation for the perturbed LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverPertGaussian<MODEL, OBS>:: applyWeights(const IncrementEnsemble4D_ & bkg_pert,
     							IncrementEnsemble4D_ & ana_pert,
							const GeometryIterator_ & i) {

	util:: Timer timer(classname(), "applyWeights");
	Log::info() << "DCC: Using applyWeights in the PERTURBED LETKF GAUSSIAN" << std::endl;

	// Loop through analysis times and ens.members:
	for (size_t itime=0; itime < bkg_pert[0].size(); ++itime) {
	    //make grid point forecast pert ensemble array:
	    Eigen::MatrixXd Xb;
	    bkg_pert.packEigen(Xb, i, itime);

	    // Postmultiply:
	    const Eigen::MatrixXd Xinc = Xb*(this->Wa_);
	    const Eigen::VectorXd xinc = Xinc.rowwise().mean();	   

	    // Generate analysis perturbations ensuring analysis perturbations centered around background perturbations:
	    Eigen::MatrixXd Xa = (Xinc + Xb).colwise() - xinc;

	    // Posterior inflation if RTPS and RTTP coefficients belong to (0,1]:
	    this->posteriorInflation(Xb, Xa);

	    // Assign Xa to ana_pert:
	    Xa.colwise() += xinc;
	    ana_pert.setEigen(Xa, i, itime); 

	} // end for loop

	Log::info() << "DCC: ApplyWeights in the PERTURBED LETKF GAUSSIAN COMPLETED" << std::endl;

} // End function applyWeights




}  // namespace oops
#endif  // OOPS_ASSIMILATION_LETKFSOLVER_H_
