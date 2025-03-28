/*
 * (C) Copyright 2018-2021 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef OOPS_RUNS_HOFX3D_H_
#define OOPS_RUNS_HOFX3D_H_

#include <string>
#include <vector>

#include "eckit/exception/Exceptions.h"
#include "oops/base/Geometry.h"
#include "oops/base/ObsAuxControls.h"
#include "oops/base/ObsErrors.h"
#include "oops/base/Observations.h"
#include "oops/base/Observers.h"
#include "oops/base/ObsSpaces.h"
#include "oops/base/PostProcessor.h"
#include "oops/base/State.h"
#include "oops/generic/instantiateObsErrorFactory.h"
#include "oops/interface/ObsDataVector.h"
#include "oops/mpi/mpi.h"
#include "oops/runs/Application.h"
#include "oops/util/DateTime.h"
#include "oops/util/Duration.h"
#include "oops/util/Logger.h"
#include "oops/util/parameters/Parameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/RequiredParameter.h"

namespace oops {

// -----------------------------------------------------------------------------

/// \brief Top-level options taken by the HofX3D application.
template <typename MODEL, typename OBS>
class HofX3DParameters : public ApplicationParameters {
  OOPS_CONCRETE_PARAMETERS(HofX3DParameters, ApplicationParameters)

  typedef Geometry<MODEL> Geometry_;
  typedef State<MODEL> State_;

 public:
  /// Options describing the assimilation time window.
  RequiredParameter<eckit::LocalConfiguration> timeWindow{"time window", this};

  /// Options describing the observations and their treatment
  RequiredParameter<eckit::LocalConfiguration> observations{"observations", this};

  /// Geometry parameters.
  RequiredParameter<eckit::LocalConfiguration> geometry{"geometry", this};

  /// Whether to save the H(x) vector as ObsValues.
  Parameter<bool> makeObs{"make obs", false, this};

  /// Initial state parameters.
  RequiredParameter<eckit::LocalConfiguration> initialCondition{"state", this};
};

// -----------------------------------------------------------------------------

/// Application computes H(x) in 3D mode from the state that is read in.
template <typename MODEL, typename OBS> class HofX3D : public Application {
  typedef Geometry<MODEL>            Geometry_;
  typedef ObsAuxControls<OBS>        ObsAux_;
  typedef Observations<OBS>          Observations_;
  typedef ObsDataVector<OBS, int>    ObsDataInt_;
  typedef ObsErrors<OBS>             ObsErrors_;
  typedef Observers<MODEL, OBS>      Observers_;
  typedef ObsSpaces<OBS>             ObsSpaces_;
  typedef State<MODEL>               State_;

  typedef HofX3DParameters<MODEL, OBS> HofX3DParameters_;

 public:
// -----------------------------------------------------------------------------
  explicit HofX3D(const eckit::mpi::Comm & comm = oops::mpi::world()) : Application(comm) {
    instantiateObsErrorFactory<OBS>();
  }
// -----------------------------------------------------------------------------
  virtual ~HofX3D() = default;
// -----------------------------------------------------------------------------
  int execute(const eckit::Configuration & fullConfig, bool validate) const override {
//  Deserialize parameters
    HofX3DParameters_ params;
    if (validate) params.validate(fullConfig);
    params.deserialize(fullConfig);

//  Setup observation window
    const util::TimeWindow timeWindow(fullConfig.getSubConfiguration("time window"));
    const util::DateTime winmidpoint = timeWindow.midpoint();
    Log::info() << "HofX3D observation window: " << timeWindow << std::endl;

//  Setup geometry
    const Geometry_ geometry(params.geometry, this->getComm(), mpi::myself());

//  Setup state
    const eckit::LocalConfiguration initialConfig(fullConfig, "state");
    State_ xx(geometry, initialConfig);
    Log::test() << "State: " << xx << std::endl;

//  Check that state is inside the obs window
    if (xx.validTime() != winmidpoint) {
      Log::error() << "State time: " << xx.validTime() << std::endl;
      Log::error() << "Obs window: " << timeWindow << std::endl;
      Log::error() << "Window midpoint: " << winmidpoint << std::endl;
      throw eckit::BadValue("The state should be valid at half of the observation window.");
    }

//  Setup observations
    const eckit::LocalConfiguration oConfig(fullConfig, "observations");
    const eckit::LocalConfiguration obsConfig(oConfig, "observers");
    ObsSpaces_ obspaces(obsConfig, this->getComm(), timeWindow);
    ObsAux_ obsaux(obspaces, obsConfig);
    ObsErrors_ Rmat(obsConfig, obspaces);

//  Setup and initialize observer
    PostProcessor<State_> post;
    Observers_ hofx(obspaces, oConfig);
    hofx.initialize(geometry, obsaux, Rmat, post);

//  Compute H(x)
    post.initialize(xx, winmidpoint, timeWindow.length());
    post.process(xx);
    post.finalize(xx);

//  Get observations from observer
    Observations_ yobs(obspaces);
    std::vector<ObsDataInt_> qcflags;
    for (size_t jj = 0; jj < obspaces.size(); ++jj) {
      ObsDataInt_ qc(obspaces[jj], obspaces[jj].obsvariables());
      qcflags.push_back(qc);
    }
    hofx.finalize(yobs, qcflags);
    Log::info() << "H(x): " << std::endl << yobs << "End H(x)" << std::endl;
    Log::test() << "H(x): " << std::endl << yobs << "End H(x)" << std::endl;

//  Perturb H(x) if needed
    if (oConfig.getBool("obs perturbations", false)) {
      yobs.perturb(Rmat);
      Log::test() << "Perturbed H(x): " << std::endl << yobs << "End Perturbed H(x)" << std::endl;
    }

//  Save H(x) as observations (if "make obs" == true)
    if (fullConfig.getBool("make obs", false)) yobs.save("ObsValue");
    obspaces.save();

    return 0;
  }
// -----------------------------------------------------------------------------
  void outputSchema(const std::string & outputPath) const override {
    HofX3DParameters_ params;
    params.outputSchema(outputPath);
  }
// -----------------------------------------------------------------------------
  void validateConfig(const eckit::Configuration & fullConfig) const override {
    HofX3DParameters_ params;
    params.validate(fullConfig);
  }
// -----------------------------------------------------------------------------
 private:
  std::string appname() const override {
    return "oops::HofX3D<" + MODEL::name() + ", " + OBS::name() + ">";
  }
// -----------------------------------------------------------------------------
};

}  // namespace oops

#endif  // OOPS_RUNS_HOFX3D_H_
