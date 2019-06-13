
struct LayerMixture{T<:Real, I<:Integer}
    μ::Vector{T}
    σ::Vector{T}
    N::Vector{T}
    k::I
end

function getpeaks(z::AbstractVector, ping::AbstractVector; thresh=0, order=1, dz=1.0)
    n = length(ping)
    D2 = DerivativeOperator{Float64}(2, order, dz, n, :Dirichlet0, :Dirichlet0)
    ii = [ci.I[1] for ci in findlocalmaxima(ping) if ping[ci] > thresh]
    curvature = D2 * ping
    jj = findall(x -> x < 0, curvature)
    ii = intersect(ii, jj)
    return z[ii], ping[ii], curvature[ii]
end

function guesslayers(z, ping; thresh=0, order=1, dz=1.0)
    μ, p, c = getpeaks(z, ping, thresh=thresh, order=order, dz=dz)
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

function refine_heights!(mix::LayerMixture, basis, z, ping)
    masked = copy(basis)
    for i in 1:mix.k
        imask = findall(z -> abs(z - mix.μ[i]) > mix.σ[i], z)
        masked[imask, i] .= 0
    end
    N1 = nnls(basis, ping)
    mix.N .= N1
end

function cost(μ, σ, N, z, ping)
    model = mixpdf(μ, σ, N, z)
    # return sum((ping.-model).^2 .* ping)
    return sum(abs2, ping - model)
    # return sum(log.(ping) .* (log.(model) .- log.(ping)))
end

function refine_σ!(mix::LayerMixture, z, ping)
    μ, σ, N, k  = mix.μ, mix.σ, mix.N, mix.k
    fit1 = optimize(θ -> cost(mix.μ, exp.(θ), N, z, ping), log.(σ))
    mix.σ[:] .= exp.(fit1.minimizer)
    return fit1
end

function refine_μσ!(mix::LayerMixture, z, ping)
    μ, σ, N, k  = mix.μ, mix.σ, mix.N, mix.k
    lx = [fill(minimum(mix.μ), mix.k); fill(-Inf, mix.k)]
    ux = [fill(maximum(mix.μ), mix.k); fill(log(25), mix.k)]
    θ = [mix.μ; log.(mix.σ)]
    fit1 = optimize(θ -> cost(θ[1:k], exp.(θ[k+1:end]), N, z, ping), θ)
    μ1 = fit1.minimizer[1:k]
    σ1 = exp.(fit1.minimizer[k+1:end])
    mix.μ[:] .= μ1
    mix.σ[:] .= σ1
    return fit1
end

function fit!(mix, z, ping; tol=eps(), trace=false)
    basis = getbasis(mix, z)
    ss = Inf
    while true
        refine_heights!(mix, basis, z, ping)
        opt = refine_σ!(mix, z, ping)
        if trace
            println(opt.minimum)
        end
        if ss - opt.minimum < tol
            return opt
        end
        ss = opt.minimum
    end
end

function fit(mix, z, ping; tol=eps(), trace=false)
    mix1 = deepcopy(mix)
    fit!(mix, z, ping; tol=eps(), trace=false)
end
