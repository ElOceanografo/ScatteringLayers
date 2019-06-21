
struct LayerMixture{T<:Real, I<:Integer}
    μ::Vector{T}
    σ::Vector{T}
    N::Vector{T}
    k::I
end

function getpeaks(ping::Ping; thresh=0, order=1, dz=1.0)
    n = length(ping)
    D2 = DerivativeOperator{Float64}(2, order, dz, n, :Dirichlet0, :Dirichlet0)
    ii = [ci.I[1] for ci in findlocalmaxima(ping) if ping[ci] > thresh]
    curvature = D2 * ping
    jj = findall(x -> x < 0, curvature)
    ii = intersect(ii, jj)
    return depths(ping)[ii], ping[ii], curvature[ii]
end

function guesslayers(ping::Ping; thresh=0, order=1, dz=1.0)
    μ, p, c = getpeaks(ping, thresh=thresh, order=order, dz=dz)
    σ = sqrt.(-p ./ c)
    N = p .* sqrt.(2pi * σ.^2)
    k = length(μ)
    return LayerMixture(μ, σ, N, k)
end

function mixpdf(μ, σ, N, z)
    res = zeros(length(z))
    k = length(μ)
    for i in 1:k
        res .+= N[i] * pdf.(Normal(μ[i], σ[i]), z)
    end
    return res
end

mixpdf(mix::LayerMixture, z) = mixpdf(mix.μ, mix.σ, mix.N, z)

function getbasis!(basis, mix, z)
    for i in 1:mix.k
        basis[:, i] = pdf.(Normal(mix.μ[i], mix.σ[i]), z)
    end
end

function getbasis(mix::LayerMixture, z::AbstractVector)
    res = zeros(length(z), mix.k)
    getbasis!(res, mix, z)
    return res
end

function refine_heights!(mix::LayerMixture, basis, ping)
    masked = copy(basis)
    for i in 1:mix.k
        imask = findall(z -> abs(z - mix.μ[i]) > mix.σ[i], depths(ping))
        masked[imask, i] .= 0
    end
    N1 = nnls(basis, ping.data)
    mix.N .= N1
end

function cost(μ, σ, N, ping)
    model = mixpdf(μ, σ, N, depths(ping))
    # return sum((ping.-model).^2 .* ping)
    return sum(abs2, ping - model)
    # return sum(log.(ping) .* (log.(model) .- log.(ping)))
end

function refine_σ!(mix::LayerMixture, ping)
    μ, σ, N, k  = mix.μ, mix.σ, mix.N, mix.k
    fit1 = optimize(θ -> cost(mix.μ, exp.(θ), N, ping), log.(σ))
    mix.σ[:] .= exp.(fit1.minimizer)
    return fit1
end

function refine_μσ!(mix::LayerMixture, ping)
    μ, σ, N, k  = mix.μ, mix.σ, mix.N, mix.k
    lx = [fill(minimum(mix.μ), mix.k); fill(-Inf, mix.k)]
    ux = [fill(maximum(mix.μ), mix.k); fill(log(25), mix.k)]
    θ = [mix.μ; log.(mix.σ)]
    fit1 = optimize(θ -> cost(θ[1:k], exp.(θ[k+1:end]), N, ping), θ)
    μ1 = fit1.minimizer[1:k]
    σ1 = exp.(fit1.minimizer[k+1:end])
    mix.μ[:] .= μ1
    mix.σ[:] .= σ1
    return fit1
end

const refine_funcs = Dict(:σ => refine_σ!, :μσ => refine_μσ!,
    :widths => refine_σ!, :widths_locs => refine_μσ!)

function fit!(mix, ping; tol=eps(), trace=false, refine=:σ)
    basis = getbasis(mix, depths(ping))
    if refine != :none
        refiner! = refine_funcs[refine]
        ss = Inf
        while true
            refine_heights!(mix, basis, ping)
            opt = refine_σ!(mix, ping)
            if trace
                println(opt.minimum)
            end
            if ss - opt.minimum < tol
                return opt
            end
            ss = opt.minimum
        end
    end
end

function fit(mix, ping; tol=eps(), trace=false, refine=:σ)
    mix1 = deepcopy(mix)
    fit!(mix1, ping; tol=eps(), trace=false)
    return mix1
end

function fitlayers(echo::Echogram; thresh=0, tol=eps(), refine=:σ)
    layers = []
    n = size(echo, 2)
    z = echo.z
    println("Detecting layers...")
    @showprogress for i in 1:n
        mix = guesslayers(getping(echo, i), thresh=thresh)
        fit!(mix, getping(echo, i), tol=tol, refine=refine)
        push!(layers, mix)
    end
    return layers
end
