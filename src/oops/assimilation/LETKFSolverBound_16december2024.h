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
		              const Eigen::MatrixXd & Yb, const Eigen::MatrixXd & HXb, 
			      const Eigen::VectorXd & diagInvR);
//                              const Eigen::MatrixXd & diagInvR);


  /// Applies weights and adds posterior inflation
  //virtual void applyWeights(const IncrementEnsemble4D_ &, IncrementEnsemble4D_ &,
  //                          const GeometryIterator_ &);

  // Protected variables:
  protected:
  DeparturesEnsemble_ HXb_;    ///< full background ensemble in the observation space HXb;
//  DeparturesEnsemble_ yb_mean_dep_;
//  Departures_ yHofx;

  // Private variables:
  private:
  // eigen solver matrices and vectors:
  Eigen::VectorXf eival_;
  Eigen::MatrixXf eivec_;

  DeparturesEnsemble_ YobsPert_;
  DeparturesEnsemble_ yb_mean_dep_;
  Departures_ yHofx;
//  Departures_ yb_mean_dep_;
//  DeparturesEnsemble_ yb_mean_dep_;
  Departures_ newR_;
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
 //   yb_mean_dep_(obspaces, this->nens_),
 //   yHofx(this->obspaces_),	
 //   yb_mean_dep_(obspaces, "ObsValue"),
    eival_(this->nens_), eivec_(this->nens_, this->nens_),
    YobsPert_(obspaces, this->nens_),
    yb_mean_dep_(obspaces, this->nens_),
    yHofx(this->obspaces_),	
    newR_(obspaces, "ObsValue"),	
    observationsConfig_(config.getSubConfiguration("observations"))
{
  Log::trace() << "LETKFSolverBound<MODEL, OBS>::create starting" << std::endl;
  Log::info()  << "Using EIGEN implementation of the PERTURBED LETKF" << std::endl;
  Log::trace() << "LETKFSolverBound<MODEL, OBS>::create done" << std::endl;
}


// -----------------------------------------------------------------------------
//  Compute HofX for the perturbed LETKF:
template <typename MODEL, typename OBS>
Observations<OBS> LETKFSolverBound<MODEL, OBS>::computeHofX(const StateEnsemble4D_ & ens_xx,
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
	    Log::info() << "yb_mean: " << yb_mean.obstodep()  << std::endl;
        }

        // Store original observation in (this->omb_):
        Observations_ yobs(this->obspaces_, "ObsValue");
        (this->omb_) = yobs.obstodep();


        Log::info() << "DCC: yobs: " << yobs << std::endl;
//        Log::info() << "DCC: yobs[1]: " << yobs[1] << std::endl;	// DOES NOT PROVIDE OBS VALUE!



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

	//
	// Computing new R:
	//
//        Departures_ yHofx(this->obspaces_);
	yHofx = (yobs.obstodep()) * (yb_mean.obstodep());	





//	Departures_ locvector0(this->obspaces_);
//	Eigen::VectorXd yobs_vec = this->omb_.packEigen(locvector0);
//      Eigen::VectorXd yb_mean_vec = yb_mean.packEigen(locvector0);

//	Log::info() << "DCC: yb_mean size: " << yb_mean_vec.size() << std::endl;	
//	for (int iobs = 0; iobs < yobs_vec.size(); ++iobs){
//	    Log::info() << "DCC: yb_mean_vec[" << iobs << "]: " << yb_mean_vec[iobs] << std::endl;
//	    yb_mean + yobs;
//	    yb_mean_vec[iobs]
//	}

	//Observations_ yb_mean(this->obspaces_);
//	(this->omb_) = yb_mean.obstodep();
//	Eigen::VectorXd ybmean_vec = this->omb_.packEigen(locvector0);

        				
//	yb_mean_vec = yb_mean.obstodep();
//	yb_mean_vec[0] = yb_mean_vec[0] * yobs_vec[0];


        return yb_mean;
}


// -----------------------------------------------------------------------------
//  Measurement update for the perturbed LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverBound<MODEL, OBS>::measurementUpdate(const IncrementEnsemble4D_ & bkg_pert,
                                                    const GeometryIterator_ & i,
                                                    IncrementEnsemble4D_ & ana_pert) {

      util::Timer timer(classname(), "measurementUpdate");
      Log::info() << "DCC: Using measurementUpdate in the BOUNDEDD LETKF" << std::endl;
//      Observations_ yb_mean(this->obspaces_);
//      Departures_ yb_mean_dep(this->obspaces_);
//      yb_mean_dep += yb_mean.obstodep();
//      Log::info() << "DCC: yb_mean_obstodep: " << yb_mean.obstodep() << std::endl;

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

      Eigen::VectorXd local_yHofx = yHofx.packEigen(locvector);
      // Computing D^(-1/2):
      Eigen::VectorXd inv_sqrt_yHofx = local_yHofx.array().sqrt().inverse();
      // Computing I + Rr^(-1):
      int p = local_omb_vec.size();
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
      Eigen::VectorXd local_invVarR_vec;
//      local_invVarR_vec = (inv_sqrt_yHofx.asDiagonal() * result * inv_sqrt_yHofx.asDiagonal()).inverse();
      local_invVarR_vec = local_R_mat.diagonal().cwiseInverse(); 
      Log::info() << "DCC: local_invVarR_vec: " << local_invVarR_vec << std::endl;
//      local_invVarR_vec = local_R_mat.inverse(); 

//      Eigen::VectorXd local_ybmean_vec = this->yb_mean.packEigen(locvector);
      //Eigen::VectorXd local_yb_mean_vec = yb_mean.packEigen(locvector);
//      Eigen::VectorXd local_yb_mean_vec = yb_mean_dep.packEigen(locvector);
//      Log::info() << "DCC: local_yb_mean: " << local_yb_mean_vec << std::endl;
//      Log::info() << "DCC: local_yb_mean size: " << local_yb_mean_vec.size() << std::endl;
//      for (int iobs = 0; iobs < local_yb_mean_vec.size(); ++iobs){
//          Log::info() << "DCC:  local_yb_mean[" << iobs << "]: " << local_yb_mean_vec[iobs] << std::endl;
//      }
      
     

      if (local_omb_vec.size() == 0) {
           Log::info() << "DCC: No observations found in this local volume. No need to update Wa_ and wa_" << std::endl;
           this->copyLocalIncrement(bkg_pert, i, ana_pert);
      } else {
              Log::info() << "DCC: Obs found in this local volume. Do normal LETKF update" << std::endl;
              Eigen::MatrixXd local_Yb_mat = (this->Yb_).packEigen(locvector); // the Eigen::MatrixXd function is used to convert the DeparturesEnsemble_ to Eigen::MatrixXd
              Eigen::MatrixXd local_YobsPert_mat = YobsPert_.packEigen(locvector);
              Eigen::MatrixXd local_HXb_mat = (this->HXb_).packEigen(locvector);
              // Create local obs errors:
//              Eigen::VectorXd local_invVarR_vec = (this->invVarR_)->packEigen(locvector);
              // Apply localization:
//              Eigen::VectorXd localization = locvector.packEigen(locvector);
//              local_invVarR_vec.array() *= localization.array();
              computeWeights(local_YobsPert_mat, local_omb_vec, local_yHofx, local_Yb_mat, local_HXb_mat, local_invVarR_vec);
//	      computeWeights(local_YobsPert_mat, local_omb_vec, local_yHofx, local_Yb_mat, local_HXb_mat, local_R_mat);
//              applyWeights(bkg_pert, ana_pert, i);
      }
}




// -----------------------------------------------------------------------------
// Compute weights for the perturbed LETKF:
template <typename MODEL, typename OBS>
void LETKFSolverBound<MODEL, OBS>::computeWeights(const Eigen::MatrixXd & YobsPert,
						 const Eigen::VectorXd & yobs,
						 const Eigen::VectorXd & yHofx,
                                                 const Eigen::MatrixXd & Yb,
                                                 const Eigen::MatrixXd & HXb,
//						 const Eigen::MatrixXd & newR){
                                                 const Eigen::VectorXd & diagInvR){

        util::Timer timer(classname(), "computeWeights");
        Log::info() << "DCC: Using computeWeights in the BOUNDED LETKF" << std::endl;
        // Compute transformation matix, save in Wa_, wa_
        const LocalEnsembleSolverInflationParameters & inflopt = (this->options_).infl;


        Eigen::MatrixXf Yb_f       = Yb.cast<float>(); // cast function converts double to float
        Eigen::MatrixXf YobsPert_f = YobsPert.cast<float>();
	Eigen::VectorXf yobs_f     = yobs.cast<float>();
        Eigen::VectorXf diagInvR_f = diagInvR.cast<float>();
//	Eigen::MatrixXf newR_f     = newR.cast<float>();
        Eigen::MatrixXf HXb_f      = HXb.cast<float>();
	Eigen::MatrixXf yHofx_f    = yHofx.cast<float>();

        Log::info() << "DCC: Yb_f: " << Yb_f << std::endl;
        Log::info() << "DCC: YobsPert_f: " << YobsPert_f << std::endl;
	Log::info() << "DCC: y_obs: " << yobs_f << std::endl;
        Log::info() << "DCC: diagInvR_f: " << diagInvR_f << std::endl;
//	Log::info() << "DCC: newR_f: " << newR_f << std::endl;
        Log::info() << "DCC: HXb_f: " << HXb_f << std::endl;
	Log::info() << "DCC: yHofx_f: " << yHofx_f << std::endl;

//	Log::info() << "DCC: compute: " << HXb_f(1,1)-Yb_f(1,1) << std::endl;
//	Eigen::MatrixXf work0;
//	Eigen::VectorXf ybmean;
//	work0 = HXb_f-Yb_f;
//	Log::info() << "DCC: work0: " <<  work0 << std::endl;
//	for (int iobs = 0; iobs < yobs_f.size(); ++iobs) {
//	     work0(1,iobs) = HXb_f(1,iobs)-Yb_f(1,iobs);   
//	     ybmean(iobs) = work0(1,iobs); // does not work
             //yb_mean_dep_[iens] = (this->HXb_)[iens] + (this->Yb_)[iens];
	     //(this->HXb_)[iens] = (this->HXb_)[iens] + (this->Yb_)[iens];
//	     Log::info() << "DCC: work0: " << work0(iobs) << std::endl;
//             Log::info() << "DCC: iobs: " << iobs << std::endl;	     
	//     Log::info() << "DCC: Yb_: " << Yb_f(iens,1) << std::endl;
//        }
	
//	Log::info() << "DCC: ybmean: " <<  ybmean << std::endl;


//        for (int iobs = 0; iobs < yobs_f.size(); ++iobs){
//            Log::info() << "DCC: iobs: " << iobs << std::endl;
//          Log::info() << "DCC: (this->R_)[" << iobs << "]: " << diagInvR_f[iobs] << std::endl;
//	    (this->R_)[iobs] = 0.23;
//            Log::info() << "DCC: (diagInvR_f)[" << iobs << "]: " << diagInvR_f[iobs] << std::endl;
//	    Log::info() << "DCC: yb_mean: " << yb_mean << std::endl;
//	    diagInvR_f[iobs] = yobs_f[iobs]+ybmean_f[iobs];
//        }

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

        //
        // Compute elementwise product of the raw observations and divide by the perturbed observations (necessary for the bounded case):
        //
//        Log::info() << "DCC: Size of the observations: " << yobs_f.size() << std::endl;
//	for (size_t iens = 0; iens < (this->nens_); ++iens) {
//            for (int iobs = 0; iobs < yobs_f.size(); ++iobs){
     	//        Log::info() << "DCC: yobs_f[" << iobs << "] : " << yobs_f[iobs] << std::endl;
//	        YobsPert_f(iens,iobs) = yobs_f[iobs] * yobs_f[iobs] / YobsPert_f(iens,iobs);
//	    }
//	}

	YobsPert_f = (yobs_f.array().square().matrix().replicate(1, this->nens_).transpose().array() / YobsPert_f.array()).matrix();
	Log::info() << "DCC: YobsPert_f: " << YobsPert_f << std::endl;


//	for (size_t iens = 0; iens < this->nens_; ++iens) {
//    	    // Vectorized computation over yobs_f for the current ensemble member
//    	    Eigen::VectorXd yobs_squared  = yobs_f.array().square();  // yobs_f[iobs] * yobs_f[iobs]
//    	    YobsPert_f.row(iens).array() = yobs_squared / YobsPert_f.row(iens).array();
//	}


//	Eigen::MatrixXf YobsPert_f2 = YobsPert.cast<float>();
//	YobsPert_f2 = (yobs_f.array().square().matrix().replicate(1, this->nens_).transpose().array() / YobsPert_f2.array()).matrix();

//	Log::info() << "DCC: YobsPert_f2: " << YobsPert_f2 << std::endl;


        // Computing Wa = Pa * (Yb^T) * R_relative^(-1) * (YobsPert_Bounded - HXb):
        Eigen::MatrixXf Wa_f(this->nens_, this->nens_);
        Wa_f = work * ( Yb_f * (diagInvR_f.asDiagonal()) * (YobsPert_f - HXb_f).transpose() );

        Log::info() << "DCC: ComputeWeights in the BOUNDED LETKF COMPLETED" << std::endl;
        this->Wa_ = Wa_f.cast<double>(); // cast function converts float to double

} // End function computeWeights





}  // namespace oops
#endif  // OOPS_ASSIMILATION_LETKFSOLVER_H_
