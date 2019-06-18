using ScatteringLayers
using CSVFiles
using DataFrames
using Dates
using DiffEqOperators
using Images
using StaticArrays


println("Testing echogram types...")
sp = SpacePoint(SVector(randn(3)...))
sp = SpacePoint(randn(3))
stp = SpaceTimePoint(SVector(randn(3)...), Dates.now())
stp = SpaceTimePoint(randn(3), Dates.now())
sps = [SpacePoint(SVector(randn(3)...)) for i in 1:10]
z = collect(1:5.0)
eg = Echogram(randn(5, 10), z, sps)


println("Loading test data...")
echo_df = DataFrame(load("data\\test_data.csv"))
fmt = DateFormat("yyyymmddHH:MM:SS.s")
echo_df[:datetime] = DateTime.(string.(echo_df[:Date_M]) .* echo_df[:Time_M], fmt)

echo = unstack(echo_df, :Layer_depth_max, :datetime, :Sv_mean)[1:end, 2:end]
echo = disallowmissing(Matrix(echo))
echo = echo[end:-1:1, :]
echo[echo .< -90] .= -90
echo = 10 .^(echo/10)
m, n = size(echo)
dz = 1.0
zz = 880.0:-dz:8.0
order = 1
D2 = DerivativeOperator{Float64}(2, order, dz, m, :Dirichlet0, :Dirichlet0)

println("Testing...")
ping = imfilter(echo, Kernel.gaussian(0.5))[:, 5]

THRESH = 10^(-80/10)
μ, p, c = getpeaks(zz, ping, thresh=THRESH)
mix = guesslayers(zz, ping, thresh=THRESH)

basis = getbasis(mix, zz)
N1 = refine_heights!(mix, basis, zz, ping)
ping_nnls = mixpdf(mix, zz)

fit1 = refine_μσ!(mix, zz, ping)
ping_refined1 = mixpdf(mix, zz)
mix = guesslayers(zz, ping, thresh=THRESH)
fit!(mix, zz, ping, trace=true)
ping_refined2 = mixpdf(mix, zz)

mix1 = fit(mix, zz, ping, trace=true)
mix1 = fit(mix, zz, ping, refine=:σ)
mix1 = fit(mix, zz, ping, refine=:σμ)
mix1 = fit(mix, zz, ping, refine=:widths)
mix1 = fit(mix, zz, ping, refine=:widths_locs)


x = SpacePoint([36.712110, -122.187028])
echogram = Echogram(imfilter(echo, Kernel.gaussian(0.5)), zz, repeat([x], n))
layers = fitlayers(echogram, thresh=THRESH)
