using StaticArrays
using Dates

abstract type AbstractSpaceTimePoint end

struct SpacePoint{S} <: AbstractSpaceTimePoint
    x::S
    SpacePoint(x::S) where {S <: SVector} = new{S}(x)
end
SpacePoint(x::AbstractVector) = SpacePoint(SVector(x...))

Base.size(sp::SpacePoint) = size(sp.x)
Base.getindex(sp::SpacePoint, i::Int) = getindex(sp.x, i)
Base.setindex!(sp::SpacePoint, v, i::Int) = setindex!(sp.x, v, i)
location(sp::SpacePoint) = sp.x
Base.show(io::IO, sp::SpacePoint) = print(io, "SpacePoint: ", location(sp).data)

struct SpaceTimePoint{S, T}
    x::S
    t::T
    function SpaceTimePoint(x::S, t::T) where {S<:SVector, T<:Dates.AbstractTime}
        return new{S, T}(x, t)
    end
end
function SpaceTimePoint(x::AbstractVector, t::T) where {S<:SVector, T<:Dates.AbstractTime}
    return SpaceTimePoint(SVector(x...), t)
end

Base.size(stp::SpaceTimePoint) = size(stp.x) + 1
Base.getindex(stp::SpaceTimePoint, i::Int) = getindex(stp.x, i)
Base.setindex!(stp::SpaceTimePoint, v, i::Int) = setindex!(stp.x, v, i)
location(stp::SpaceTimePoint) = stp.x

function Base.show(io::IO, ::MIME"text/plain", sp::SpaceTimePoint)
    print(io, "SpaceTimePoint: ", location(stp).data, ", ", sp.t)
end
Base.show(io::IO, stp::SpaceTimePoint) = print(io, "SpaceTimePoint: ", location(stp).data,", ",  stp.t)

# function Base.show(io::IO, ::MIME"text/plain", sp::SpacePoint{S}) where {S}
#     print(sp)
# end


struct Echogram{T,N,S} <: AbstractArray{T,N}
    data::Array{T,N}
    z::AbstractVector{T}
    s::AbstractVector{S}
end

Base.size(eg::Echogram) = size(eg.data)
Base.getindex(eg::Echogram, i) = getindex(eg.data, i)
Base.getindex(eg::Echogram{T,N}, inds...) where {T,N} = getindex(eg.data, inds...)
# Base.getindex(eg::Echogram{T,N}, I::Vararg{Int, N}) where {T,N} = getindex(eg.data, I)
Base.setindex!(eg::Echogram, i) = getindex(eg.data, i)
Base.setindex!(eg::Echogram{T,N}, v, I::Vararg{Int, N}) where {T,N} = setindex!(eg.data, v, I)

depths(eg::Echogram) = eg.z

function Base.show(io::IO, ::MIME"text/plain", eg::Echogram{T,N,S}) where {T,N,S}
    println(io, "Echogram{$T,$N}")
    println(io, "Depth: $(minimum(eg.z)):$(maximum(eg.z))")
    println(io, "Locations: $(first(eg.s))...$(last(eg.s))")
    show(io, MIME"text/plain"(), eg.data)
end
