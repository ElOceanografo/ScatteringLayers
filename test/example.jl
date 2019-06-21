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
heatmap(10log10.(imfilter(echo, Kernel.gaussian((2, 1)))))
# echo = imfilter(echo, Kernel.gaussian((1, 1)))

ping = imfilter(echo, Kernel.gaussian((2, 1)))[:, 10]
zz = Float64.(880:-1:8)
deimos_x = SpacePoint([36.712110, -122.187028])
ping = Ping(ping, zz, [deimos_x])

THRESH = 10^(-90/10)
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

echogram = Echogram(imfilter(echo, Kernel.gaussian((2, 1))), zz, repeat([deimos_x], n))
THRESH = -90
layers = fitlayers(echogram, thresh=THRESH, refine=:none)
[l.k for l in layers]

p = heatmap(1:length(layers), -zz, 10log10.(echo), color=:grayscale, clim=(-90, NaN))
# p = heatmap(1:length(layers), -zz, echo, clim=(1e-8, 4e-7))
for i in 1:length(layers)
    scatter!(p, i*ones(layers[i].k), -layers[i].μ,
        color=:red, markeralpha=1, markerstrokecolor=:red, markersize=1e6*layers[i].N, legend=false)
end
p
savefig(p, "layerdots.png")


using LightGraphs
using MetaGraphs

function pushlayers!(g, mix, time)
    for i in 1:mix.k
        dict = Dict(:μ=>mix.μ[i], :σ=>mix.σ[i], :N=>mix.N[i], :k=>mix.k, :time=>time, :v=>0)
        add_vertex!(g, dict)
    end
end

function getnearest(x1, x2)
    n1, n2 = length(x1), length(x2)
    nearest = zeros(Int, n1)
    distances = similar(x1)
    for i in 1:n1
        dist = Inf
        for j in 1:n2
            dist1 = abs(x1[i] - x2[j])
            if dist1 < dist
                dist = dist1
                nearest[i] = j
                distances[i] = dist
            end
        end
    end
    return nearest, distances
end

@time nearest1, dists = getnearest(layers[1].μ, layers[2].μ)

using ProgressMeter
function link(layers, times, max_dist=Inf, max_rel_change=Inf, thresh=0)
    g = MetaDiGraph()
    # Add all layers to graph
    for (i, mix) in enumerate(layers)
        pushlayers!(g, mix, times[i])
    end
    t1 = times[1]
    printstyled("Linking...\n", color=:green)
    @showprogress for i in 2:length(times)
        t2 = times[i]
        v1 = collect(filter_vertices(g, :time, t1))
        v2 = collect(filter_vertices(g, :time, t2))
        μ1 = [get_prop(g, v, :μ) for v in v1]
        μ2 = [get_prop(g, v, :μ) for v in v2]
        N1 = [get_prop(g, v, :N) for v in v1]
        N2 = [get_prop(g, v, :N) for v in v2]
        nearest_μ, dists_μ = getnearest(μ1, μ2)
        nearest_N, dists_N = getnearest(N1, N2)

        N_order = sortperm(N1, rev=true)
        for j in N_order
            d = dists_μ[j]
            if (d <= max_dist) &
                (dists_N[j] / N1[j] < max_rel_change) &
                (length(inneighbors(g, v2[nearest_μ[j]])) == 0)
                add_edge!(g, v1[j], v2[nearest_μ[j]])
            end
        end
        t1 = t2
    end
    return g
end

g = link(layers, 1:420, 30, 2)

gr(size=(1800, 1000))
p = heatmap(1:length(layers), -zz, 10log10.(echo), color=:grayscale,
    clim=(THRESH, NaN))
for e in edges(g)
    v1 = src(e)
    v2 = dst(e)
    x1 = get_prop(g, v1, :μ)
    t1 = get_prop(g, v1, :time)
    x2 = get_prop(g, v2, :μ)
    t2 = get_prop(g, v2, :time)
    if get_prop(g, v1, :N) > 10^(-75/10)
        plot!(p, [t1, t2], -[x1, x2], color=:red, legend=false)
    end
end
p


μ = [get_prop(g, v, :μ) for v in vertices(g)]
σ = [get_prop(g, v, :σ) for v in vertices(g)]
N = [get_prop(g, v, :N) for v in vertices(g)]
using StatsPlots
histogram2d(μ, N, nbins=200)
histogram2d(log10.(N), log10.(σ), nbins=200, xlabel="N", ylabel="sigma")
