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
  // eigen solver matrices and vectors:
  Eigen::VectorXf eival_;
  Eigen::MatrixXf eivec_;

  DeparturesEnsemble_ YobsPert_;
  eckit::LocalConfiguration observationsConfig_; // Configuration for observations


};


// -----------------------------------------------------------------------------
//  Constructor of the LETKF solver:
template <typename MODEL, typename OBS>
LETKFSolverPert<MODEL, OBS>::LETKFSolverPert(ObsSpaces_ & obspaces, const Geometry_ & geometry,
                                     const eckit::Configuration & config, size_t nens,
                                     const State4D_ & xbmean, const Variables & incvars)
  : LETKFSolver<MODEL, OBS>(obspaces, geometry, config, nens, xbmean, incvars),
    HXb_(obspaces, this->nens_),
    eival_(this->nens_), eivec_(this->nens_, this->nens_),
    YobsPert_(obspaces, this->nens_),
    observationsConfig_(config.getSubConfiguration("observations"))
{
  Log::trace() << "LETKFSolverPert<MODEL, OBS>::create starting" << std::endl;
  Log::info()  << "Using EIGEN implementation of the PERTURBED LETKF" << std::endl;
  Log::trace() << "LETKFSolverPert<MODEL, OBS>::create done" << std::endl;
}





// -----------------------------------------------------------------------------
//  Compute HofX for the perturbed LETKF:
template <typename MODEL, typename OBS>
Observations<OBS> LETKFSolverPert<MODEL, OBS>::computeHofX(const StateEnsemble4D_ & ens_xx,
		                                          size_t iteration, bool readFromFile) {
        util::Timer timer(classname(), "computeHofX");

	Observations_ yb_mean(this->obspaces_);
        yb_mean = LETKFSolver<MODEL, OBS>::computeHofX(ens_xx, iteration, readFromFile);
	Log::info() << "----------------------------------" << std::endl;	
	Log::info() << "DCC: COMPUTING HofX" << std::endl;
	Log::info() << "DCC: this->nens_ : " << this->nens_ << std::endl;


        // Recover the full HofX from the original ensemble and store them in (this->Yb_):
	// Remember: Yb_ stands for ensemble perturbations in the observation space: Yb_ = HXb - Hxb
        for (size_t iens = 0; iens < (this->nens_); ++iens) {
	    HXb_[iens] = yb_mean + (this->Yb_)[iens];   
        }

	// Store original observation in (this->omb_):
        Observations_ yobs(this->obspaces_, "ObsValue");
	(this->omb_) = yobs.obstodep();

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
//  Measurement update for the perturbed LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverPert<MODEL, OBS>::measurementUpdate(const IncrementEnsemble4D_ & bkg_pert,
		                                    const GeometryIterator_ & i,
						    IncrementEnsemble4D_ & ana_pert) {

      util::Timer timer(classname(), "measurementUpdate");
      Log::info() << "DCC: Using measurementUpdate in the PERTURBED LETKF" << std::endl;
      
      // Create the local subset of observations:
      Departures_ locvector(this->obspaces_);
      locvector.ones();
      this->obsloc().computeLocalization(i, locvector);
      for (size_t iens = 0; iens < (this->nens_); ++iens) {
	   Log::info() << "DCC: iens: " << iens << std::endl;
	   (this->invVarR_)->mask(this->HXb_[iens]);
      }
      locvector.mask(*(this->invVarR_));
      Eigen::VectorXd local_omb_vec = this->omb_.packEigen(locvector);

      if (local_omb_vec.size() == 0) {
	   Log::info() << "DCC: No observations found in this local volume. No need to update Wa_ and wa_" << std::endl;
	   this->copyLocalIncrement(bkg_pert, i, ana_pert);
      } else { 	      
	      Log::info() << "DCC: Obs found in this local volume. Do normal LETKF update" << std::endl;
	      Eigen::MatrixXd local_Yb_mat = (this->Yb_).packEigen(locvector); // the Eigen::MatrixXd function is used to convert the DeparturesEnsemble_ to Eigen::MatrixXd
	      Eigen::MatrixXd local_YobsPert_mat = YobsPert_.packEigen(locvector);
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
void LETKFSolverPert<MODEL, OBS>::computeWeights(const Eigen::MatrixXd & YobsPert,
						 const Eigen::MatrixXd & Yb,
						 const Eigen::MatrixXd & HXb,
						 const Eigen::VectorXd & diagInvR){

	util::Timer timer(classname(), "computeWeights");
        Log::info() << "DCC: Using computeWeights in the PERTURBED LETKF" << std::endl;
	// Compute transformation matix, save in Wa_, wa_
	const LocalEnsembleSolverInflationParameters & inflopt = (this->options_).infl;
        
        Eigen::MatrixXf Yb_f       = Yb.cast<float>(); // cast function converts double to float
	Eigen::MatrixXf YobsPert_f = YobsPert.cast<float>();
	Eigen::VectorXf diagInvR_f = diagInvR.cast<float>();
	Eigen::MatrixXf HXb_f      = HXb.cast<float>();

	Log::info() << "DCC: Yb_f: " << Yb_f << std::endl;
	Log::info() << "DCC: YobsPert_f: " << YobsPert_f << std::endl;
	Log::info() << "DCC: diagInvR_f: " << diagInvR_f << std::endl;
	Log::info() << "DCC: HXb_f: " << HXb_f << std::endl;

	// Compute work = (Yb^T) R^(-1) Yb + [(nens-1)/infl I]
	const float infl = inflopt.mult;
	Eigen::MatrixXf work = Yb_f*(diagInvR_f.asDiagonal()*Yb_f.transpose());
	work.diagonal() += Eigen::VectorXf::Constant(this->nens_, (this->nens_-1)/infl);

	// Compute eigenvalues and eigenvectors of the above matrix:
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(work);
	eival_ = es.eigenvalues().real();
	eivec_ = es.eigenvectors().real();

	// Computing Pa = [ (Yb^T) R^(-1) Yb + (nens-1)/infl I  ]^(-1):
	work = eivec_ * (eival_.cwiseInverse().asDiagonal()) * eivec_.transpose(); // cwiseInverse() computes the inverse of the eigenvalues
	
	// Computing Wa = Pa * (Yb^T) * R^(-1) * (YobsPert - HXb):
	Eigen::MatrixXf Wa_f(this->nens_, this->nens_);
	Wa_f = work * ( Yb_f * (diagInvR_f.asDiagonal()) * (YobsPert_f - HXb_f).transpose() );

	Log::info() << "DCC: ComputeWeights in the PERTURBED LETKF COMPLETED" << std::endl;
	this->Wa_ = Wa_f.cast<double>(); // cast function converts float to double

} // End function computeWeights


// -----------------------------------------------------------------------------
// Apply weights and add posterior inflation for the perturbed LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverPert<MODEL, OBS>:: applyWeights(const IncrementEnsemble4D_ & bkg_pert,
     						IncrementEnsemble4D_ & ana_pert,
						const GeometryIterator_ & i) {

	util:: Timer timer(classname(), "applyWeights");
	Log::info() << "DCC: Using applyWeights in the PERTURBED LETKF" << std::endl;

	// Allocating matrices:
	Eigen::MatrixXd XbOriginal;

	// Loop through analysis times and ens.members:
	for (size_t itime=0; itime < bkg_pert[0].size(); ++itime) {
	    //make grid point forecast pert ensemble array:
	    bkg_pert.packEigen(XbOriginal, i, itime);

	    // Postmultiply:
	    Eigen::MatrixXd Xa = XbOriginal*(this->Wa_);
	    Eigen::VectorXd xa = Xa.rowwise().mean();	   

	    // Generate analysis perturbations for inflation:
	    Eigen::MatrixXd XaRecenter = (Xa + XbOriginal).colwise() - xa;

	    // Posterior inflation if RTPS and RTTP coefficients belong to (0,1]:
	    this->posteriorInflation(XbOriginal, XaRecenter);

	    // Assign Xa to ana_pert:
	    Xa = XaRecenter.colwise() + xa;
	    ana_pert.setEigen(Xa, i, itime); 

	} // end for loop

	Log::info() << "DCC: ApplyWeights in the PERTURBED LETKF COMPLETED" << std::endl;

} // End function applyWeights




}  // namespace oops
#endif  // OOPS_ASSIMILATION_LETKFSOLVER_H_
