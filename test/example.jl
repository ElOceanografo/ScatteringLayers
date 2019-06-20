using Plots
pyplot(size=(1800, 1000))
using FileIO
using DataFrames
using Images
using Dates
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

heatmap(10log10.(echo))
heatmap(imfilter(echo, Kernel.gaussian(0.75)), clim=(0, 1e-7))
# echo = imfilter(echo, Kernel.gaussian((1, 1)))

ping = imfilter(echo, Kernel.gaussian(0.5))[:, 10]
zz = Float64.(880:-1:8)
deimos_x = SpacePoint([36.712110, -122.187028])
ping = Ping(ping, zz, [deimos_x])

THRESH = 10^(-80/10)
μ, p, c = getpeaks(ping, thresh=THRESH)
mix = guesslayers(ping, thresh=THRESH)
plot(zz, ping, label="Ping")
scatter!(μ, p, label="Peaks")
# plot!(zz, mixpdf(mix, zz), label="Naive")

basis = getbasis(mix, zz)
N1 = refine_heights!(mix, basis, ping)
ping_nnls = mixpdf(mix, zz)
plot!(zz, ping_nnls, label="NNLS")

fit1 = refine_μσ!(mix, ping)
ping_refined1 = mixpdf(mix, zz)
plot!(zz, ping_refined1, label="Refined  1")
mix = guesslayers(ping, thresh=THRESH)
fit!(mix, ping)
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
fit!(mix1, ping)
plot!(zz, mixpdf(mix1, zz), color=:black, alpha=0.2)

echogram = Echogram(imfilter(echo, Kernel.gaussian(0.5)), zz, repeat([deimos_x], n))
layers = fitlayers(echogram, thresh=THRESH)
[l.k for l in layers]

# p = heatmap(1:length(layers), -zz, 10log10.(echo))
p = heatmap(1:length(layers), -zz, echo, clim=(1e-8, 4e-7))
for i in 1:length(layers)
    scatter!(p, i*ones(layers[i].k), -layers[i].μ,
        color=:blue, markeralpha=1, markerstrokecolor=:blue, markersize=1e6*layers[i].N, legend=false)
end
p



using NearestNeighbors
using LightGraphs
using MetaGraphs
using Distances


g = MetaDiGraph(3)
set_props!(g, 1, Dict(:time => 0.5))
set_props!(g, 2, Dict(:time => 0.75))
set_props!(g, 3, Dict(:time => 0.75))
add_edge!(g, 1, 2)

g

collect(filter_vertices(g, :time, 0.5))
collect(filter_vertices(g, :time, 0.75))

neighbors(g, 2)
props(g, 1)

function link(layers, times)
    nverts = sum(mix.k for mix in layers)
    g = MetaDiGraph(nverts)
    for (i, mix) in enumerate(layers)
        dict = Dict(:μ => mix.μ, :σ => mix.σ, :N => mix.N, :k => mix.k, :time => times[i])
        add_vertex!(g, dict)
    end
