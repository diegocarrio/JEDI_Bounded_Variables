/*
 * (C) Copyright 2009-2016 ECMWF.
 * (C) Crown Copyright 2023, the Met Office.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef OOPS_RUNS_GENENSPERTB_H_
#define OOPS_RUNS_GENENSPERTB_H_

#include <memory>
#include <sstream>
#include <string>

#include "eckit/config/LocalConfiguration.h"
#include "oops/base/Geometry.h"
#include "oops/base/Increment.h"
#include "oops/base/instantiateCovarFactory.h"
#include "oops/base/Model.h"
#include "oops/base/ModelSpaceCovarianceBase.h"
#include "oops/base/ParameterTraitsVariables.h"
#include "oops/base/PostProcessor.h"
#include "oops/base/State.h"
#include "oops/base/State4D.h"
#include "oops/base/StateWriter.h"
#include "oops/base/Variables.h"
#include "oops/interface/ModelAuxControl.h"
#include "oops/mpi/mpi.h"
#include "oops/runs/Application.h"
#include "oops/util/ConfigHelpers.h"
#include "oops/util/DateTime.h"
#include "oops/util/Duration.h"
#include "oops/util/Logger.h"

namespace oops {

/// Options taken by the GenEnsPertB application.
template <typename MODEL> class GenEnsPertBParameters : public ApplicationParameters {
  OOPS_CONCRETE_PARAMETERS(GenEnsPertBParameters, ApplicationParameters)

 public:
  typedef State<MODEL>                                 State_;
  typedef ModelAuxControl<MODEL>                       ModelAux_;

  /// Geometry parameters.
  RequiredParameter<eckit::LocalConfiguration> geometry{"geometry", this};

  /// Model parameters.
  RequiredParameter<eckit::LocalConfiguration> model{"model", this};

  /// Initial state parameters.
  RequiredParameter<eckit::LocalConfiguration> initialCondition{"initial condition", this};

  /// Augmented model state.
  Parameter<eckit::LocalConfiguration> modelAuxControl{"model aux control",
                                                       eckit::LocalConfiguration(), this};

  /// Forecast length.
  RequiredParameter<util::Duration> forecastLength{"forecast length", this};

  /// List of variables to perturb.
  RequiredParameter<Variables> perturbedVariables{"perturbed variables", this};

  /// Background error covariance model.
  RequiredParameter<eckit::LocalConfiguration> backgroundError{"background error", this};

  /// Size of the perturbed ensemble to generate.
  RequiredParameter<int> members{"members", this};

  /// Where to write the output.
  RequiredParameter<eckit::LocalConfiguration> output{"output", this};

  /// Whether to include the control as member zero
  Parameter<bool> includeControl{"include control", false, this};
};

// -----------------------------------------------------------------------------

template <typename MODEL> class GenEnsPertB : public Application {
  typedef ModelSpaceCovarianceBase<MODEL>           CovarianceBase_;
  typedef CovarianceFactory<MODEL>                  CovarianceFactory_;
  typedef Geometry<MODEL>                           Geometry_;
  typedef Model<MODEL>                              Model_;
  typedef ModelAuxControl<MODEL>                    ModelAux_;
  typedef Increment<MODEL>                          Increment_;
  typedef Increment4D<MODEL>                        Increment4D_;
  typedef State<MODEL>                              State_;
  typedef State4D<MODEL>                            State4D_;

  typedef GenEnsPertBParameters<MODEL>              GenEnsPertBParameters_;

 public:
// -----------------------------------------------------------------------------
  explicit GenEnsPertB(const eckit::mpi::Comm & comm = oops::mpi::world()) : Application(comm) {
    instantiateCovarFactory<MODEL>();
  }
// -----------------------------------------------------------------------------
  virtual ~GenEnsPertB() {}
// -----------------------------------------------------------------------------
  int execute(const eckit::Configuration & fullConfig, bool validate) const override {
//  Deserialize parameters
    GenEnsPertBParameters_ params;
    if (validate) params.validate(fullConfig);
    params.deserialize(fullConfig);

//  Setup resolution
    const Geometry_ resol(params.geometry, this->getComm(), oops::mpi::myself());

//  Setup Model
    const eckit::LocalConfiguration modelConfig(fullConfig, "model");
    const Model_ model(resol, modelConfig);

//  Setup initial state
    const State4D_ xx(resol, eckit::LocalConfiguration(fullConfig, "initial condition"));
    Log::test() << "Initial state: " << xx << std::endl;

//  Setup augmented state
    const ModelAux_ moderr(resol, fullConfig.getSubConfiguration("model aux control"));

//  Setup times
    const util::Duration fclength(fullConfig.getString("forecast length"));
    const util::DateTime bgndate(xx[0].validTime());
    const util::DateTime enddate(bgndate + fclength);
    Log::info() << "Running forecast from " << bgndate << " to " << enddate << std::endl;

//  Setup variables
    const Variables vars(fullConfig, "perturbed variables");

//  Setup B matrix
    const eckit::LocalConfiguration covConf(fullConfig, "background error");
    std::unique_ptr<CovarianceBase_> Bmat(CovarianceFactory_::create(
                                            resol, vars, covConf, xx, xx));

    if (fullConfig.getBool("include control", false)) {
//    Save control as ensemble member 0
      State_ xp(xx[0]);

//    Setup forecast outputs
      PostProcessor<State_> post;

      eckit::LocalConfiguration outConfig(fullConfig, "output");
      util::setMember(outConfig, 0);

      post.enrollProcessor(new StateWriter<State_>(outConfig));

//    Run forecast
      model.forecast(xp, moderr, fclength, post);
      Log::test() << " Control Member final state: " << xp << std::endl;
    }

//  Generate perturbed states
    const int members = fullConfig.getInt("members");
    for (int jm = 0; jm < members; ++jm) {
//    Initial incremental state
      Increment4D_ dx(resol, vars, xx.times());

//    Generate pertubation
      Bmat->randomize(dx);

//    Add mean state
      State_ xp(xx[0]);
      xp += dx[0];

//    Setup forecast outputs
      PostProcessor<State_> post;

      eckit::LocalConfiguration outConfig(fullConfig, "output");
      util::setMember(outConfig, jm+1);

      post.enrollProcessor(new StateWriter<State_>(outConfig));

//    Run forecast
      model.forecast(xp, moderr, fclength, post);
      Log::test() << "Member " << jm << " final state: " << xp << std::endl;
    }

    return 0;
  }
// -----------------------------------------------------------------------------
  void outputSchema(const std::string & outputPath) const override {
    GenEnsPertBParameters_ params;
    params.outputSchema(outputPath);
  }
// -----------------------------------------------------------------------------
  void validateConfig(const eckit::Configuration & fullConfig) const override {
    GenEnsPertBParameters_ params;
    params.validate(fullConfig);
  }
// -----------------------------------------------------------------------------
 private:
  std::string appname() const override {
    return "oops::GenEnsPertB<" + MODEL::name() + ">";
  }
// -----------------------------------------------------------------------------
};

}  // namespace oops
#endif  // OOPS_RUNS_GENENSPERTB_H_
