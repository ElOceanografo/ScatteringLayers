module ScatteringLayers

using Optim
using NNLS
using Distributions
using DiffEqOperators
using Images

export AbstractSpaceTimePoint,
    SpacePoint,
    SpaceTimePoint,
    Echogram,
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
