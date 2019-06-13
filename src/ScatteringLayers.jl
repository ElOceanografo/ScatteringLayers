module ScatteringLayers

using Optim
using NNLS
using Distributions
using DiffEqOperators
using Images

export LayerMixture,
    getpeaks,
    guesslayers,
    mixpdf,
    getbasis!,
    getbasis,
    refine_heights!,
    refine_σ!,
    refine_μσ!,
    fit!

include("layerfitting.jl")

end # module
