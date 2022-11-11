module FullStaggeredGrid

using Plots, Printf, LinearAlgebra, SpecialFunctions, CairoMakie, Makie.GeometryBasics
using ExtendableSparse
import SparseArrays:spdiagm

# Macros
@views    ∂_∂x(f1,f2,Δx,Δy,∂ξ∂x,∂η∂x) = ∂ξ∂x.*(f1[2:size(f1,1),:] .- f1[1:size(f1,1)-1,:]) ./ Δx .+ ∂η∂x.*(f2[:,2:size(f2,2)] .- f2[:,1:size(f2,2)-1]) ./ Δy
@views    ∂_∂y(f1,f2,Δx,Δy,∂ξ∂y,∂η∂y) = ∂ξ∂y.*(f2[2:size(f2,1),:] .- f2[1:size(f2,1)-1,:]) ./ Δx .+ ∂η∂y.*(f1[:,2:size(f1,2)] .- f1[:,1:size(f1,2)-1]) ./ Δy
@views    ∂_∂(fE,fW,fN,fS,Δ,a,b) = a*(fE - fW) / Δ.x .+ b*(fN - fS) / Δ.y
@views    ∂_∂1(∂f∂ξ,∂f∂η, a,b) = a*∂f∂ξ .+ b*∂f∂η
@views   avWESN(A,B)  = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])

include("FSG_Assembly.jl")
export AssembleKuuKupKpu!

include("FSG_Rheology.jl")
export DevStrainRateStressTensor!

include("FSG_Residual.jl")
export LinearMomentumResidual!

include("FSG_Visu.jl")
export PatchPlotMakie

end
