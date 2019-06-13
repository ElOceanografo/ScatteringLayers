using DiffEqOperators
using Distributions
using Optim
using NNLS
using Plots
pyplot(size=(1800, 1000))

using ScatteringLayers

echo_df = DataFrame(load("test\\data\\deimos-2019-04.csv"))
fmt = DateFormat("yyyymmddHH:MM:SS.s")
echo_df[:datetime] = DateTime.(string.(echo_df[:Date_M]) .* echo_df[:Time_M], fmt)

echo = unstack(echo_df, :Layer_depth_max, :datetime, :Sv_mean)[3:end, 2:end]
echo = disallowmissing(Matrix(echo))
echo = echo[end:-1:1, :]
echo[echo .< -90] .= -90
echo = 10 .^(echo/10)
m, n = size(echo)
dz = 1.0
order = 1
D2 = DerivativeOperator{Float64}(2, order, dz, m, :Dirichlet0, :Dirichlet0)

heatmap(10log10.(echo))
heatmap(imfilter(echo, Kernel.gaussian(0.75)), clim=(0, 1e-7))
# echo = imfilter(echo, Kernel.gaussian((1, 1)))

ping = imfilter(echo, Kernel.gaussian(0.5))[:, 10]
zz = Float64.(880:-1:8)

THRESH = 10^(-80/10)
μ, p, c = getpeaks(zz, ping, thresh=THRESH)
mix = guesslayers(zz, ping, thresh=THRESH)
plot(zz, ping, label="Ping")
scatter!(μ, p, label="Peaks")
# plot!(zz, mixpdf(mix, zz), label="Naive")

basis = getbasis(mix, zz)
N1 = refine_heights!(mix, basis, zz, ping)
ping_nnls = mixpdf(mix, zz)
plot!(zz, ping_nnls, label="NNLS")

fit1 = refine_μσ!(mix, zz, ping)
ping_refined1 = mixpdf(mix, zz)
plot!(zz, ping_refined1, label="Refined  1")
mix = guesslayers(zz, ping, thresh=THRESH)
fit!(mix, zz, ping)
ping_refined2 = mixpdf(mix, zz)
plot!(zz, ping_refined2, label="Refined 2")
scatter!(mix.μ, mixpdf(mix, mix.μ))


plot(cumsum(abs2.(ping .- ping_nnls)))
plot!(cumsum(abs2.(ping .- ping_refined1)))
plot!(cumsum(abs2.(ping .- ping_refined2)))

plot(zz, 10log10.(ping), label="Ping", ylim=(-90, -60))
plot!(zz, 10log10.(basis * N1), label="NNLS")
plot!(zz, 10log10.(mixpdf(mix, zz)), label="Refined σ")


plot(zz, ping, legend=false)
mix1 = deepcopy(mix)
mix1.μ .+= randn()
mix1.σ .+= max(0, randn())
mix1.N .+= 1e-9randn()
fit!(mix1, zz, ping)
plot!(zz, mixpdf(mix1, zz), color=:black, alpha=0.2)
