/*
 * (C) Copyright 2020 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */


#ifndef OOPS_ASSIMILATION_LETKFSOLVERBOUNDED_H_
#define OOPS_ASSIMILATION_LETKFSOLVERBOUNDED_H_

#include <Eigen/Dense>
#include <cfloat>
#include <memory>
#include <string>
#include <vector>

#include "eckit/config/Configuration.h"
#include "oops/assimilation/LETKFSolver.h"
#include "oops/assimilation/LETKFSolverPert.h"
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
 * An implementation of the Bounded LETKF.
 */

template <typename MODEL, typename OBS>
class LETKFSolverBound : public LETKFSolver<MODEL, OBS> {
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
  static const std::string classname() {return "oops::LETKFSolverBounded";}

  LETKFSolverBound(ObsSpaces_ &, const Geometry_ &, const eckit::Configuration &, size_t,
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
  virtual void computeWeights(const Eigen::MatrixXd & YobsPert, const Eigen::VectorXd & yobs,
		              const Eigen::VectorXd & kk,
			      const Eigen::MatrixXd & Yb, const Eigen::VectorXd & ybmean, 
//		              const Eigen::MatrixXd & Yb, const Eigen::MatrixXd & HXb, 
			      const Eigen::VectorXd & diagInvR);
//                              const Eigen::MatrixXd & diagInvR);


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
  Departures_ yHofx_;
  Departures_ zeroDepartures_;
  Departures_ yb_mean_;
  Departures_ yobs_;
  Observations_ zeroObservations_;
  Observations_ yobs;
//  DeparturesEnsemble_ zero_;
  eckit::LocalConfiguration observationsConfig_; // Configuration for observations


};


// -----------------------------------------------------------------------------
//  Constructor of the Bounded LETKF solver:
template <typename MODEL, typename OBS>
LETKFSolverBound<MODEL, OBS>::LETKFSolverBound(ObsSpaces_ & obspaces, const Geometry_ & geometry,
                                     const eckit::Configuration & config, size_t nens,
                                     const State4D_ & xbmean, const Variables & incvars)
  : LETKFSolver<MODEL, OBS>(obspaces, geometry, config, nens, xbmean, incvars),
    HXb_(obspaces, this->nens_),
    eival_(this->nens_), eivec_(this->nens_, this->nens_),
    YobsPert_(obspaces, this->nens_),
    yHofx_(this->obspaces_),
    zeroDepartures_(this->obspaces_),
    yb_mean_(this->obspaces_),	
    yobs_(this->obspaces_),
//  zero_(this->obspaces_, this->nens_),	
    zeroObservations_(this->obspaces_),
    yobs(this->obspaces_, "ObsValue"),
    observationsConfig_(config.getSubConfiguration("observations"))
{
  Log::trace() << "LETKFSolverBound<MODEL, OBS>::create starting" << std::endl;
  Log::info()  << "Using EIGEN implementation of the BOUNDED LETKF" << std::endl;
  Log::trace() << "LETKFSolverBound<MODEL, OBS>::create done" << std::endl;
}


// -----------------------------------------------------------------------------
//  Compute HofX for the bounded LETKF:
template <typename MODEL, typename OBS>
Observations<OBS> LETKFSolverBound<MODEL, OBS>::computeHofX(const StateEnsemble4D_ & ens_xx,
                                                          size_t iteration, bool readFromFile) {
        util::Timer timer(classname(), "computeHofX");

        //Observations_ yb_mean(this->obspaces_);
        Observations_ yb_mean = LETKFSolver<MODEL, OBS>::computeHofX(ens_xx, iteration, readFromFile);
//	this->yb_mean_ = yb_mean;  // Sergey Suggestion. DOES NOT WORK!
	this->yb_mean_ = yb_mean - zeroObservations_;

        Log::info() << "----------------------------------" << std::endl;
        Log::info() << "DCC: COMPUTING HofX" << std::endl;

        // Recover the full HofX from the original ensemble and store them in (this->Yb_):
        // Remember: Yb_ stands for ensemble perturbations in the observation space: Yb_ = HXb - Hxb
        for (size_t iens = 0; iens < (this->nens_); ++iens) {
            HXb_[iens] = yb_mean + (this->Yb_)[iens];
	    Log::info() << "HXb_[" << iens << "]: " << HXb_[iens] << std::endl;
        }

        // Store original observation:
//	Departures_ yobs(this->obspaces_);
	this->yobs_ = yobs - zeroObservations_;
//	this->yobs = yobs_; // Sergey Suggestion. DOES NOT WORK!

//      Observations_ yobs(this->obspaces_, "ObsValue");
//      (this->omb_) = yobs.obstodep();
//      (this->omb_) = yobs + zero_;  //avoid storing yobs in this->omb
	

        // Loop through ensemble to generate perturbed observations:
        //
        Departures_ yobsPertSum(this->obspaces_); // Create object yobsPertSum that will contain the sum of the perturbed observations
        for (size_t iens = 0; iens < (this->nens_); ++iens) {
            Observations_ ypert(yobs);
            ypert.perturb(*(this->R_));
//          YobsPert_[iens] = ypert.obstodep();
	    YobsPert_[iens] = ypert + zeroDepartures_;
            YobsPert_[iens].mask(this->Yb_[iens]);
            yobsPertSum += YobsPert_[iens];
        }

        //
        // Recenter the perturbed observations around original obs:
        //
        for (size_t iens = 0; iens < (this->nens_); ++iens) {
            YobsPert_[iens].axpy(-1.0/(this->nens_), yobsPertSum);
//          YobsPert_[iens] += yobs.obstodep();
	    YobsPert_[iens] = yobs + YobsPert_[iens];
        }

	//
	// Computing elementwise product between observations y and the mean of ensemble in observation space yb_mean:
	//
//	yHofx = (yobs.obstodep()) * (yb_mean.obstodep());
	yHofx_ = yobs - zeroObservations_;
//	yHofx_ *= yb_mean;  // Sergey Suggestion. DOES NOT WORK!
	yHofx_ *= this->yb_mean_;

        return yb_mean;  // Do we need this????
}


// -----------------------------------------------------------------------------
//  Measurement update for the bounded LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverBound<MODEL, OBS>::measurementUpdate(const IncrementEnsemble4D_ & bkg_pert,
                                                    const GeometryIterator_ & i,
                                                    IncrementEnsemble4D_ & ana_pert) {

      util::Timer timer(classname(), "measurementUpdate");
      Log::info() << "DCC: Using measurementUpdate in the BOUNDEDD LETKF" << std::endl;

      // Create the local subset of observations:
      Departures_ locvector(this->obspaces_);
      locvector.ones();
      this->obsloc().computeLocalization(i, locvector);
      for (size_t iens = 0; iens < (this->nens_); ++iens) {
           Log::info() << "DCC: iens: " << iens << std::endl;
           (this->invVarR_)->mask(this->HXb_[iens]);
      }
      locvector.mask(*(this->invVarR_));
      Eigen::VectorXd local_yobs_vec = this->yobs_.packEigen(locvector);  // DOES NOT WORK!!!
//      Eigen::VectorXd local_yobs_vec = this->yobs.packEigen(locvector);  // DOES NOT WORK!!!
      Eigen::VectorXd local_ybmean_vec = this->yb_mean_.packEigen(locvector);

      Eigen::VectorXd local_yHofx = yHofx_.packEigen(locvector);
      // Computing D^(-1/2):
      Eigen::VectorXd inv_sqrt_yHofx = local_yHofx.array().sqrt().inverse();
      // Computing I + Rr^(-1):
      int p = local_yobs_vec.size();
      double Rr = 0.25;
      double inv_Rr = 1.0 / Rr;
      Log::info() << "p: " << p << std::endl;
      Eigen::MatrixXd I = Eigen::MatrixXd::Identity(p, p);
      Eigen::MatrixXd Rr_inv = Eigen::MatrixXd::Identity(p, p) * inv_Rr;
      Eigen::MatrixXd result = I + Rr_inv;
      Log::info() << "DCC: I + Rr^(-1): " << result << std::endl;
      // Computing new R for the bounded variables:
      Eigen::MatrixXd local_R_mat;
      local_R_mat = inv_sqrt_yHofx.asDiagonal() * result * inv_sqrt_yHofx.asDiagonal();
      Log::info() << "DCC: local_R_mat: " << local_R_mat << std::endl;
      // Computing inverse of diagonal elements of R:
      Eigen::VectorXd local_invVarR_vec;
      local_invVarR_vec = local_R_mat.diagonal().cwiseInverse(); 
      Log::info() << "DCC: local_invVarR_vec: " << local_invVarR_vec << std::endl;


      if (local_yobs_vec.size() == 0) {
           Log::info() << "DCC: No observations found in this local volume. No need to update Wa_ and wa_" << std::endl;
           this->copyLocalIncrement(bkg_pert, i, ana_pert);
      } else {
              Log::info() << "DCC: Obs found in this local volume. Do normal LETKF update" << std::endl;
              Eigen::MatrixXd local_Yb_mat = (this->Yb_).packEigen(locvector); // the Eigen::MatrixXd function is used to convert the DeparturesEnsemble_ to Eigen::MatrixXd
	      // Compute HXb:
//	      Eigen::MatrixXd local_HXb_mat;
//	      local_HXb_mat = local_ybmean_vec + local_Yb_mat;                         // IS THIS CORRECT??????
              Eigen::MatrixXd local_YobsPert_mat = YobsPert_.packEigen(locvector);
//              Eigen::MatrixXd local_HXb_mat = (this->HXb_).packEigen(locvector);
//            // Create local obs errors:
//            Eigen::VectorXd local_invVarR_vec = (this->invVarR_)->packEigen(locvector);
//            // Apply localization:
//            Eigen::VectorXd localization = locvector.packEigen(locvector);
//            local_invVarR_vec.array() *= localization.array();
//            // Computing weights:  
//              computeWeights(local_YobsPert_mat, local_yobs_vec, local_yHofx, local_Yb_mat, local_HXb_mat, local_invVarR_vec);
              computeWeights(local_YobsPert_mat, local_yobs_vec, local_yHofx, local_Yb_mat, local_ybmean_vec, local_invVarR_vec);
	      // Applying weights:
              applyWeights(bkg_pert, ana_pert, i);
      }
}




// -----------------------------------------------------------------------------
// Compute weights for the Bounded LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverBound<MODEL, OBS>::computeWeights(const Eigen::MatrixXd & YobsPert,
						 const Eigen::VectorXd & yobs,
						 const Eigen::VectorXd & yHofx,
                                                 const Eigen::MatrixXd & Yb,
//                                                 const Eigen::MatrixXd & HXb,
						 const Eigen::VectorXd & ybmean,
                                                 const Eigen::VectorXd & diagInvR){

        util::Timer timer(classname(), "computeWeights");
        Log::info() << "DCC: Using computeWeights in the BOUNDED LETKF" << std::endl;
        // Compute transformation matix, save in Wa_, wa_
        const LocalEnsembleSolverInflationParameters & inflopt = (this->options_).infl;

        Eigen::MatrixXf Yb_f       = Yb.cast<float>(); // cast function converts double to float
        Eigen::MatrixXf YobsPert_f = YobsPert.cast<float>();
	Eigen::VectorXf yobs_f     = yobs.cast<float>();
        Eigen::VectorXf diagInvR_f = diagInvR.cast<float>();
//      Eigen::MatrixXf HXb_f      = HXb.cast<float>();
	Eigen::VectorXf ybmean_f   = ybmean.cast<float>();
	Eigen::MatrixXf yHofx_f    = yHofx.cast<float>();

        Log::info() << "DCC: Yb_f: " << Yb_f << std::endl;
        Log::info() << "DCC: YobsPert_f: " << YobsPert_f << std::endl;
	Log::info() << "DCC: y_obs: " << yobs_f << std::endl;
        Log::info() << "DCC: diagInvR_f: " << diagInvR_f << std::endl;
//      Log::info() << "DCC: HXb_f: " << HXb_f << std::endl;
	Log::info() << "DCC: ybmean_f: " << ybmean_f << std::endl;
	Log::info() << "DCC: yHofx_f: " << yHofx_f << std::endl;

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

        // Compute elementwise product of the raw observations and divide by the perturbed observations (necessary for the bounded case):
	YobsPert_f = (yobs_f.array().square().matrix().replicate(1, this->nens_).transpose().array() / YobsPert_f.array()).matrix();
	Log::info() << "DCC: YobsPert_f: " << YobsPert_f << std::endl;

	// Computing HXb:
	Eigen::MatrixXf HXb_f;
	HXb_f = (ybmean_f.array().matrix().replicate(1, this->nens_).transpose().array() + Yb_f.array()).matrix();
        Log::info() << "DCC: HXb_f: " << HXb_f << std::endl;
	Log::info() << "DCC: ybmean matrix: " << ybmean_f.array().matrix().replicate(1, this->nens_).transpose().array().matrix() << std::endl;

        // Computing Wa = Pa * (Yb^T) * R_relative^(-1) * (YobsPert_Bounded - HXb):
        Eigen::MatrixXf Wa_f(this->nens_, this->nens_);
        Wa_f = work * ( Yb_f * (diagInvR_f.asDiagonal()) * (YobsPert_f - HXb_f).transpose() );

        Log::info() << "DCC: ComputeWeights in the BOUNDED LETKF COMPLETED" << std::endl;
        this->Wa_ = Wa_f.cast<double>(); // cast function converts float to double

} // End function computeWeights



// -----------------------------------------------------------------------------
// Apply weights and add posterior inflation for the Bounded LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverBound<MODEL, OBS>:: applyWeights(const IncrementEnsemble4D_ & bkg_pert,
                                                IncrementEnsemble4D_ & ana_pert,
                                                const GeometryIterator_ & i) {

        util:: Timer timer(classname(), "applyWeights");
        Log::info() << "DCC: Using applyWeights in the BOUNDED LETKF" << std::endl;

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

        Log::info() << "DCC: ApplyWeights in the BOUNDED LETKF COMPLETED" << std::endl;

} // End function applyWeights




}  // namespace oops
#endif  // OOPS_ASSIMILATION_LETKFSOLVERBOUNDED__
