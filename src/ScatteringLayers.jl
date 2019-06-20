module ScatteringLayers

using Optim
using NNLS
using Distributions
using DiffEqOperators
using Images
using ProgressMeter

export AbstractSpaceTimePoint,
    SpacePoint,
    SpaceTimePoint,
    Echogram,
    Ping,
    depths,
    spacetime,
    getping,
    LayerMixture,
    getpeaks,
    guesslayers,
    mixpdf,
    getbasis!,
    getbasis,
    refine_heights!,
    refine_σ!,
    refine_μσ!,
    fit!,
    fit,
    fitlayers

include("echograms.jl")
include("layerfitting.jl")

end # module
