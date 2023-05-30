# example triad 2D kernel
using TinyKernels, Printf, GLMakie, Makie.GeometryBasics
import LinearAlgebra: norm
import Statistics: mean
Makie.inline!(false)

@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end
@views av1D(x) = 0.5.*(x[2:end] .+ x[1:end-1])
@views avWESN(A,B) = 0.25*(A[1:end-1,:] .+ A[2:end-0,:] .+ B[:,1:end-1] .+ B[:,2:end-0])
@views ∂_∂ξ(aE, aW, Δ) = (aE - aW) / Δ.x
@views ∂_∂η(aN, aS, Δ) = (aN - aS) / Δ.y

# add coordinate transformation
# ix --> i, iy -->j
# Introduce local time stepping
# PERFECT SCALING ON CPU AND FLOAT64!

include("DataStructures_v2.jl")
include("setup_example.jl")

# Select based upon your local device (:CPU, :CUDA, :AMDGPU, :Metal)
backend = :CPU

include("helpers.jl") # will be defined in TinyKernels soon

@setup_example()

###############################

function PatchPlotMakie(vertx, verty, sol, xmin, xmax, ymin, ymax; cmap = :turbo, write_fig=false )
    f   = Figure(resolution = (1200, 1000))

    ar = (xmax - xmin) / (ymax - ymin)

    Axis(f[1,1])
    min_v = minimum( sol.p ); max_v = maximum( sol.p )

    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.p)]
    poly!(p, color = sol.p, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # Axis(f[2,1], aspect = ar)
    # min_v = minimum( sol.vx ); max_v = maximum( sol.vx )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    # poly!(p, color = sol.vx, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # Axis(f[2,2], aspect = ar)
    # min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    # poly!(p, color = sol.vy, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    
    # scatter!(x1,y1, color=:white)
    # scatter!(x2,y2, color=:white, marker=:xcross)
    Colorbar(f[1, 2], colormap = cmap, limits=limits, flipaxis = true, size = 25 )

    display(f)
    # if write_fig==true 
    #     FileIO.save( string(@__DIR__, "/plot.png"), f)
    # end
    return nothing
end

###############################

@tiny function StrainRates!(ε̇, ∇v, V, ∂, Δ)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    @inbounds if isin(∇v.x)
        ∂ξ∂x = ∂.ξ.∂x.x[i,j]; ∂ξ∂y = ∂.ξ.∂y.x[i,j]; ∂η∂x = ∂.η.∂x.x[i,j]; ∂η∂y = ∂.η.∂y.x[i,j];
        ∂Vx∂ξ       = ∂_∂ξ(V.x.c[i+1,j+1], V.x.c[i+0,j+1], Δ) 
        ∂Vy∂η       = ∂_∂η(V.y.v[i+0,j+1], V.y.v[i+0,j+0], Δ)
        ∂Vx∂η       = ∂_∂η(V.x.v[i+0,j+1], V.x.v[i+0,j+0], Δ)
        ∂Vy∂ξ       = ∂_∂ξ(V.y.c[i+1,j+1], V.y.c[i+0,j+1], Δ)
        ∂Vx∂x       = ∂Vx∂ξ * ∂ξ∂x + ∂Vx∂η * ∂η∂x
        ∂Vy∂y       = ∂Vy∂ξ * ∂ξ∂y + ∂Vy∂η * ∂η∂y
        ∂Vx∂y       = ∂Vx∂ξ * ∂ξ∂y + ∂Vx∂η * ∂η∂y
        ∂Vy∂x       = ∂Vy∂ξ * ∂ξ∂x + ∂Vy∂η * ∂η∂x
        ∇v.x[i,j]   = ∂Vx∂x + ∂Vy∂y
        ε̇.xx.x[i,j] = ∂Vx∂x - 1//3*(∂Vx∂x + ∂Vy∂y)
        ε̇.yy.x[i,j] = ∂Vy∂y - 1//3*(∂Vx∂x + ∂Vy∂y)
        ε̇.xy.x[i,j] = 1//2*(∂Vx∂y + ∂Vy∂x)
    end
    @inbounds if isin(∇v.y)
        ∂ξ∂x = ∂.ξ.∂x.y[i,j]; ∂ξ∂y = ∂.ξ.∂y.y[i,j]; ∂η∂x = ∂.η.∂x.y[i,j]; ∂η∂y = ∂.η.∂y.y[i,j];
        ∂Vx∂ξ       = ∂_∂ξ(V.x.v[i+1,j+0], V.x.v[i+0,j+0], Δ)
        ∂Vy∂η       = ∂_∂η(V.y.c[i+1,j+1], V.y.c[i+1,j+0], Δ)
        ∂Vx∂η       = ∂_∂η(V.x.c[i+1,j+1], V.x.c[i+1,j+0], Δ)
        ∂Vy∂ξ       = ∂_∂ξ(V.y.v[i+1,j+0], V.y.v[i+0,j+0], Δ)
        ∂Vx∂x       = ∂Vx∂ξ * ∂ξ∂x + ∂Vx∂η * ∂η∂x
        ∂Vy∂y       = ∂Vy∂ξ * ∂ξ∂y + ∂Vy∂η * ∂η∂y
        ∂Vx∂y       = ∂Vx∂ξ * ∂ξ∂y + ∂Vx∂η * ∂η∂y
        ∂Vy∂x       = ∂Vy∂ξ * ∂ξ∂x + ∂Vy∂η * ∂η∂x
        ∇v.y[i,j]   = ∂Vx∂x + ∂Vy∂y
        ε̇.xx.y[i,j] = ∂Vx∂x - 1//3*(∂Vx∂x + ∂Vy∂y)
        ε̇.yy.y[i,j] = ∂Vy∂y - 1//3*(∂Vx∂x + ∂Vy∂y)
        ε̇.xy.y[i,j] = 1//2*(∂Vx∂y + ∂Vy∂x)
    end
end

@tiny function Stress!(τ, η, ε̇)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    @inbounds if isin(τ.xx.x)
        τ.xx.x[i,j] = 2 * η.x[i,j] * ε̇.xx.x[i,j]
        τ.yy.x[i,j] = 2 * η.x[i,j] * ε̇.yy.x[i,j]
        τ.xy.x[i,j] = 2 * η.x[i,j] * ε̇.xy.x[i,j]
    end
    @inbounds if isin(τ.xx.y)
        τ.xx.y[i,j] = 2 * η.y[i,j] * ε̇.xx.y[i,j]
        τ.yy.y[i,j] = 2 * η.y[i,j] * ε̇.yy.y[i,j]
        τ.xy.y[i,j] = 2 * η.y[i,j] * ε̇.xy.y[i,j]
    end
end

@tiny function Residuals!(R, τ, P, ∂, b, Δ)
    i, j = @indices
    @inbounds if i>1 && j>1 && i<size(R.x.v,1) && j<size(R.x.v,2)
        ∂ξ∂x = ∂.ξ.∂x.v[i,j]; ∂ξ∂y = ∂.ξ.∂y.v[i,j]; ∂η∂x = ∂.η.∂x.v[i,j]; ∂η∂y = ∂.η.∂y.v[i,j];
        ∂τxx∂ξ     = ∂_∂ξ(τ.xx.y[i,j], τ.xx.y[i-1,j], Δ)
        ∂τxx∂η     = ∂_∂η(τ.xx.y[i,j], τ.xx.y[i,j-1], Δ)
        ∂τyy∂ξ     = ∂_∂ξ(τ.yy.y[i,j], τ.yy.y[i-1,j], Δ)
        ∂τyy∂η     = ∂_∂η(τ.yy.x[i,j], τ.yy.x[i,j-1], Δ)
        ∂τyx∂η     = ∂_∂η(τ.xy.x[i,j], τ.xy.x[i,j-1], Δ)
        ∂τxy∂ξ     = ∂_∂ξ(τ.xy.y[i,j], τ.xy.y[i-1,j], Δ)
        ∂P∂ξ       = ∂_∂ξ(P.y[i,j],    P.y[i-1,j],    Δ)
        ∂P∂η       = ∂_∂η(P.x[i,j],    P.x[i,j-1],    Δ)
        ∂τxx∂x     = ∂τxx∂ξ * ∂ξ∂x + ∂τxx∂η * ∂η∂x
        ∂τyy∂y     = ∂τyy∂ξ * ∂ξ∂y + ∂τyy∂η * ∂η∂y 
        ∂τyx∂y     = ∂τxy∂ξ * ∂ξ∂y + ∂τyx∂η * ∂η∂y            # ⚠ assume τyx = τxy !!
        ∂τxy∂x     = ∂τxy∂ξ * ∂ξ∂x + ∂τyx∂η * ∂η∂x            # ⚠ assume τyx = τxy !!
        ∂P∂x       = ∂P∂ξ   * ∂ξ∂x + ∂P∂η   * ∂η∂x
        ∂P∂y       = ∂P∂ξ   * ∂ξ∂y + ∂P∂η   * ∂η∂y 
        R.x.v[i,j] = ∂τxx∂x + ∂τyx∂y - ∂P∂x
        R.y.v[i,j] = ∂τyy∂y + ∂τxy∂x - ∂P∂y
    end
    @inbounds if i>1 && j>1 && i<size(R.x.c,1) && j<size(R.x.c,2)
        ∂ξ∂x = ∂.ξ.∂x.c[i,j]; ∂ξ∂y = ∂.ξ.∂y.c[i,j]; ∂η∂x = ∂.η.∂x.c[i,j]; ∂η∂y = ∂.η.∂y.c[i,j];
        ∂τxx∂ξ     = ∂_∂ξ(τ.xx.x[i,j-1], τ.xx.x[i-1,j-1], Δ)
        ∂τxx∂η     = ∂_∂η(τ.xx.y[i-1,j], τ.xx.y[i-1,j-1], Δ)
        ∂τyy∂ξ     = ∂_∂ξ(τ.yy.x[i,j-1], τ.yy.x[i-1,j-1], Δ)
        ∂τyy∂η     = ∂_∂η(τ.yy.y[i-1,j], τ.yy.y[i-1,j-1], Δ)
        ∂τyx∂η     = ∂_∂η(τ.xy.y[i-1,j], τ.xy.y[i-1,j-1], Δ)
        ∂τxy∂ξ     = ∂_∂ξ(τ.xy.x[i,j-1], τ.xy.x[i-1,j-1], Δ)
        ∂P∂ξ       = ∂_∂ξ(P.x[i,j-1],    P.x[i-1,j-1],    Δ)
        ∂P∂η       = ∂_∂η(P.y[i-1,j],    P.y[i-1,j-1],    Δ)
        ∂τxx∂x     = ∂τxx∂ξ * ∂ξ∂x + ∂τxx∂η * ∂η∂x
        ∂τyy∂y     = ∂τyy∂ξ * ∂ξ∂y + ∂τyy∂η * ∂η∂y 
        ∂τyx∂y     = ∂τxy∂ξ * ∂ξ∂y + ∂τyx∂η * ∂η∂y            # assume τyx = τxy
        ∂τxy∂x     = ∂τxy∂ξ * ∂ξ∂x + ∂τyx∂η * ∂η∂x            # assume τyx = τxy
        ∂P∂x       = ∂P∂ξ   * ∂ξ∂x + ∂P∂η   * ∂η∂x
        ∂P∂y       = ∂P∂ξ   * ∂ξ∂y + ∂P∂η   * ∂η∂y 
        R.x.c[i,j] = ∂τxx∂x + ∂τyx∂y - ∂P∂x
        R.y.c[i,j] = ∂τyy∂y + ∂τxy∂x - ∂P∂y
    end
end

@tiny function RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ)
    i, j = @indices
    @inbounds if i>1 && j>1 && i<size(R.x.v,1) && j<size(R.x.v,2)
        ∂V∂τ.x.v[i,j] = (1-θ) * ∂V∂τ.x.v[i,j] + R.x.v[i,j]
        ∂V∂τ.y.v[i,j] = (1-θ) * ∂V∂τ.y.v[i,j] + R.y.v[i,j]
    end 
    @inbounds if i>1 && j>1 && i<size(R.x.c,1) && j<size(R.x.c,2)
        ∂V∂τ.x.c[i,j] = (1-θ) * ∂V∂τ.x.c[i,j] + R.x.c[i,j]
        ∂V∂τ.y.c[i,j] = (1-θ) * ∂V∂τ.y.c[i,j] + R.y.c[i,j]
    end
    @inline isin(A) = checkbounds(Bool, A, i, j)
    @inbounds if isin(∂P∂τ.x)
        ∂P∂τ.x[i,j] = -∇v.x[i,j]
    end
    @inbounds if isin(∂P∂τ.y)
        ∂P∂τ.y[i,j] = -∇v.y[i,j]
    end
end

@tiny function SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP)
    i, j = @indices
    @inbounds if i>1 && j>1 && i<size(V.x.v,1) && j<size(V.x.v,2)
        V.x.v[i,j] += ΔτV.v[i,j] * ∂V∂τ.x.v[i,j]
        V.y.v[i,j] += ΔτV.v[i,j] * ∂V∂τ.y.v[i,j]
    end 
    @inbounds if i>1 && j>1 && i<size(V.x.c,1) && j<size(V.x.c,2)
        V.x.c[i,j] += ΔτV.c[i,j] * ∂V∂τ.x.c[i,j]
        V.y.c[i,j] += ΔτV.c[i,j] * ∂V∂τ.y.c[i,j]
    end
    @inline isin(A) = checkbounds(Bool, A, i, j)
    @inbounds if isin(P.x)
        P.x[i,j] += ΔτP.x[i,j] * ∂P∂τ.x[i,j]
    end
    @inbounds if isin(P.y)
        P.y[i,j] += ΔτP.y[i,j] * ∂P∂τ.y[i,j]
    end
end

function Mesh_x( X, A, x0, σ, b, m, xmin0, xmax0, σx )
    xmin1 = (sinh.( σx.*(xmin0.-x0) ))
    xmax1 = (sinh.( σx.*(xmax0.-x0) ))
    sx    = (xmax0-xmin0)/(xmax1-xmin1)
    x     = (sinh.( σx.*(X[1].-x0) )) .* sx  .+ x0  
    # x = X[1]       
    return x
end

function Mesh_y( X, A, x0, σ, b, m, xmin0, xmax0, σx )
    xmin1 = (sinh.( σx.*(xmin0.-x0) ))
    xmax1 = (sinh.( σx.*(xmax0.-x0) ))
    sx    = (xmax0-xmin0)/(xmax1-xmin1)
    x     = (sinh.( σx.*(X[2].-x0) )) .* sx  .+ x0   
    # x = X[2]     
    return x
end

function ComputeForwardTransformation_ini!( ∂x, ∂y, x_ini, y_ini, X_msh, Amp, x0, σ, m, xmin, xmax, ymin, ymax, σx, σy, ϵ)
 
    @time for i in eachindex(y_ini)          
    
        # compute dxdksi
        X_msh[1] = x_ini[i]-ϵ
        X_msh[2] = y_ini[i] 
        xm       = Mesh_x( X_msh,  Amp, x0, σ, xmax, m, xmin, xmax, σx )
        # --------
        X_msh[1] = x_ini[i]+ϵ
        X_msh[2] = y_ini[i]
        xp       = Mesh_x( X_msh,  Amp, x0, σ, xmax, m, xmin, xmax, σx )
        # --------
        ∂x.∂ξ[i] = (xp - xm) / (2ϵ)
    
        # compute dydeta
        X_msh[1] = x_ini[i]
        X_msh[2] = y_ini[i]-ϵ
        xm     = Mesh_x( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy )
        # --------
        X_msh[1] = x_ini[i]
        X_msh[2] = y_ini[i]+ϵ
        xp       = Mesh_x( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy )
        # --------
        ∂x.∂η[i] = (xp - xm) / (2ϵ)
    
        # compute dydksi
        X_msh[1] = x_ini[i]-ϵ
        X_msh[2] = y_ini[i] 
        ym       = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy )
        # --------
        X_msh[1] = x_ini[i]+ϵ
        X_msh[2] = y_ini[i]
        yp       = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy )
        # --------
        ∂y.∂ξ[i] = (yp - ym) / (2ϵ)
    
        # compute dydeta
        X_msh[1] = x_ini[i]
        X_msh[2] = y_ini[i]-ϵ
        ym     = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy )
        # --------
        X_msh[1] = x_ini[i]
        X_msh[2] = y_ini[i]+ϵ
        yp     = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy )
        # --------
        ∂y.∂η[i] = (yp - ym) / (2ϵ)
    end
    # #################
    # # ForwardDiff
    # g = zeros(2)
    # Y = zeros(1)
    # dydksi_FD = zeros(size(dydeta))
    # dydeta_FD = zeros(size(dydeta))
    # dxdksi_FD = zeros(size(dydeta))
    # dxdeta_FD = zeros(size(dydeta))
    # @time for i in eachindex(dydeta_FD)
    #     X_msh[1] = x_ini[i]
    #     X_msh[2] = y_ini[i]
    #     Mesh_y_closed = (X_msh) -> Mesh_y( X_msh, Amp, x0, σ, b, m, ymin )
    #     ForwardDiff.gradient!( g, Mesh_y_closed, X_msh )
    #     dydksi_FD[i] = g[1]
    #     dydeta_FD[i] = g[2]
    #     Meshx_surf_closed = (X_msh) -> Mesh_x( X_msh, Amp, x0, σ, b, m, ymin )
    #     ForwardDiff.gradient!( g, Meshx_surf_closed, X_msh )
    #     dxdksi_FD[i] = g[1]
    #     dxdeta_FD[i] = g[2]
    # end
    
    # dxdksi_num = diff(xv4,dims=1)/(Δx/2)
    # dxdeta_num = diff(xv4,dims=2)/(Δy/2)
    # dydksi_num = diff(yv4,dims=1)/(Δx/2)
    # dydeta_num = diff(yv4,dims=2)/(Δy/2)
    
    # @printf("min(dxdksi    ) = %1.6f --- max(dxdksi    ) = %1.6f\n", minimum(dxdksi   ), maximum(dxdksi   ))
    # @printf("min(dxdksi_FD ) = %1.6f --- max(dxdksi_FD ) = %1.6f\n", minimum(dxdksi_FD), maximum(dxdksi_FD))
    # @printf("min(dxdksi_num) = %1.6f --- max(dxdksi_num) = %1.6f\n", minimum(dxdksi_num), maximum(dxdksi_num))
    
    # @printf("min(dxdeta    ) = %1.6f --- max(dxdeta   ) = %1.6f\n", minimum(dxdeta   ), maximum(dxdeta   ))
    # @printf("min(dxdeta_FD ) = %1.6f --- max(dxdeta_FD) = %1.6f\n", minimum(dxdeta_FD), maximum(dxdeta_FD))
    # @printf("min(dxdeta_num) = %1.6f --- max(dxdeta_num) = %1.6f\n", minimum(dxdeta_num), maximum(dxdeta_num))
    
    # @printf("min(dydksi    ) = %1.6f --- max(dydksi    ) = %1.6f\n", minimum(dydksi   ), maximum(dydksi   ))
    # @printf("min(dydksi_FD ) = %1.6f --- max(dydksi_FD ) = %1.6f\n", minimum(dydksi_FD), maximum(dydksi_FD))
    # @printf("min(dydksi_num) = %1.6f --- max(dydksi_num) = %1.6f\n", minimum(dydksi_num), maximum(dydksi_num))
    
    # @printf("min(dydeta    ) = %1.6f --- max(dydeta    ) = %1.6f\n", minimum(dydeta   ), maximum(dydeta   ))
    # @printf("min(dydeta_FD ) = %1.6f --- max(dydeta_FD ) = %1.6f\n", minimum(dydeta_FD), maximum(dydeta_FD))
    # @printf("min(dydeta_num) = %1.6f --- max(dydeta_num) = %1.6f\n", minimum(dydeta_num), maximum(dydeta_num))
    return nothing
end

##########################################
    
function InverseJacobian!(∂ξ,∂η,∂x,∂y)
    M = zeros(2,2)
    @time for i in eachindex(∂ξ.∂x)
        M[1,1]   = ∂x.∂ξ[i]
        M[1,2]   = ∂x.∂η[i]
        M[2,1]   = ∂y.∂ξ[i]
        M[2,2]   = ∂y.∂η[i]
        invJ     = inv(M)
        ∂ξ.∂x[i] = invJ[1,1]
        ∂ξ.∂y[i] = invJ[1,2]
        ∂η.∂x[i] = invJ[2,1]
        ∂η.∂y[i] = invJ[2,2]
    end
    @printf("min(∂ξ∂x) = %1.6f --- max(∂ξ∂x) = %1.6f\n", minimum(∂ξ.∂x), maximum(∂ξ.∂x))
    @printf("min(∂ξ∂y) = %1.6f --- max(∂ξ∂y) = %1.6f\n", minimum(∂ξ.∂y), maximum(∂ξ.∂y))
    @printf("min(∂η∂x) = %1.6f --- max(∂η∂x) = %1.6f\n", minimum(∂η.∂x), maximum(∂η.∂x))
    @printf("min(∂η∂y) = %1.6f --- max(∂η∂y) = %1.6f\n", minimum(∂η.∂y), maximum(∂η.∂y))
    return nothing
end


###############################

function main(::Type{DAT}; device) where DAT
    n          = 1
    ncx, ncy   = n*120-2, n*120-2
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    ε̇bg        = -1
    rad        = 0.5
    ϵ          = 5e-8      # nonlinear tolerence
    itmax      = 20000     # max number of iters
    errx0      = (1. , 1.0)
    nout       = 500       # check frequency
    θ          = 3.0/ncx
    η_inc      = 1e3
    adapt_mesh = true
    adapt_mesh = false

    if adapt_mesh
        cflV       = 0.25/4
        cflP       = 1.0*2
    else
        cflV       = 0.25
        cflP       = 1.0
    end

    η    = ScalarFSG(DAT, device, :XY, ncx, ncy)
    P    = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ΔτP  = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ΔτV  = ScalarFSG(DAT, device, :CV, ncx, ncy)
    ∇v   = ScalarFSG(DAT, device, :XY, ncx, ncy)
    τ    = TensorFSG(DAT, device, :XY, ncx, ncy)
    ε̇    = TensorFSG(DAT, device, :XY, ncx, ncy)
    V    = VectorFSG(DAT, device, ncx, ncy)
    R    = VectorFSG(DAT, device, ncx, ncy)
    b    = VectorFSG(DAT, device, ncx, ncy)
    ∂V∂τ = VectorFSG(DAT, device, ncx, ncy)
    ∂P∂τ = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ∂    = JacobianFSG(DAT, device, ncx, ncy)

    # Preprocessing
    Δ   = (; x =(xmax-xmin)/ncx, y =(ymax-ymin)/ncy)
    xce = (; x=LinRange(xmin-Δ.x/2, xmax+Δ.x/2, ncx+2), y=LinRange(ymin-Δ.y/2, ymax+Δ.y/2, ncy+2) ) 
    xv  = (; x=av1D(xce.x), y=av1D(xce.y) )
    xc  = (; x=av1D(xv.x),  y=av1D(xv.y) ) 
    
    xxv, yyv    = LinRange(xmin-Δ.x/2, xmax+Δ.x/2, 2ncx+3), LinRange(ymin-Δ.y/2, ymax+Δ.y/2, 2ncy+3)
    (xv4, yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])

    if adapt_mesh
        x0     = (xmax + xmin)/2
        m      = ymin
        Amp    = 1.0
        σ      = 0.5
        σx     = 0.5
        σy     = 0.5
        ϵjac   = 1e-6
        # copy initial y
        ξv     = copy(xv4)
        ηv     = copy(yv4)
        X_msh  = zeros(2)
        # Compute slope
        # hx     = -dhdx.(ξ, Amp, σ, ymax, x0)
        # Deform mesh
        for i in eachindex(ξv)          
            X_msh[1] = ξv[i]
            X_msh[2] = ηv[i]     
            xv4[i]   = Mesh_x( X_msh,  Amp, x0, σ, ymax, m, xmin, xmax, σx )
            yv4[i]   = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy )
        end
        # # Compute forward transformation
        # params = (Amp=Amp, x0=x0, σ=σ, m=m, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, σx=σx, σy=σy, ϵ=ϵ)
        ∂x     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ∂y     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ComputeForwardTransformation_ini!( ∂x, ∂y, ξv, ηv, X_msh, Amp, x0, σ, m, xmin, xmax, ymin, ymax, σx, σy, ϵjac)
        # Solve for inverse transformation
        ∂ξ = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4))); ∂η = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4)))
        InverseJacobian!(∂ξ,∂η,∂x,∂y)
        # ∂ξ∂x .= ∂ξ.∂x; ∂ξ∂y .= ∂ξ.∂y
        # ∂η∂x .= ∂η.∂x; ∂η∂y .= ∂η.∂y
        ∂.ξ.∂x.x .= to_device( ∂ξ.∂x[2:2:end-1,3:2:end-2])
        ∂.ξ.∂x.y .= to_device( ∂ξ.∂x[3:2:end-2,2:2:end-1])
        ∂.ξ.∂x.c .= to_device( ∂ξ.∂x[1:2:end-0,1:2:end-0])
        ∂.ξ.∂x.v .= to_device( ∂ξ.∂x[2:2:end-1,2:2:end-1])
        ∂.ξ.∂y.x .= to_device( ∂ξ.∂y[2:2:end-1,3:2:end-2])
        ∂.ξ.∂y.y .= to_device( ∂ξ.∂y[3:2:end-2,2:2:end-1])
        ∂.ξ.∂y.c .= to_device( ∂ξ.∂y[1:2:end-0,1:2:end-0])
        ∂.ξ.∂y.v .= to_device( ∂ξ.∂y[2:2:end-1,2:2:end-1])
        ∂.η.∂x.x .= to_device( ∂η.∂x[2:2:end-1,3:2:end-2])
        ∂.η.∂x.y .= to_device( ∂η.∂x[3:2:end-2,2:2:end-1])
        ∂.η.∂x.c .= to_device( ∂η.∂x[1:2:end-0,1:2:end-0])
        ∂.η.∂x.v .= to_device( ∂η.∂x[2:2:end-1,2:2:end-1])
        ∂.η.∂y.x .= to_device( ∂η.∂y[2:2:end-1,3:2:end-2])
        ∂.η.∂y.y .= to_device( ∂η.∂y[3:2:end-2,2:2:end-1])
        ∂.η.∂y.c .= to_device( ∂η.∂y[1:2:end-0,1:2:end-0])
        ∂.η.∂y.v .= to_device( ∂η.∂y[2:2:end-1,2:2:end-1])
    else
        # Coordinate transformation
        ∂.ξ.∂x.x .= to_device( ones(ncx+1, ncy  ))
        ∂.ξ.∂x.y .= to_device( ones(ncx,   ncy+1))
        ∂.ξ.∂x.c .= to_device( ones(ncx+2, ncy+2))
        ∂.ξ.∂x.v .= to_device( ones(ncx+1, ncy+1))
        ∂.η.∂y.x .= to_device( ones(ncx+1, ncy  ))
        ∂.η.∂y.y .= to_device( ones(ncx,   ncy+1))
        ∂.η.∂y.c .= to_device( ones(ncx+2, ncy+2))
        ∂.η.∂y.v .= to_device( ones(ncx+1, ncy+1))
    end

    X = (v=(x=xv4[2:2:end-1,2:2:end-1], y=yv4[2:2:end-1,2:2:end-1]), c=(x=xv4[1:2:end-0,1:2:end-0], y=yv4[1:2:end-0,1:2:end-0]), x=(x=xv4[2:2:end-1,3:2:end-2], y=yv4[2:2:end-1,3:2:end-2]), y=(x=xv4[3:2:end-2,2:2:end-1], y=yv4[3:2:end-2,2:2:end-1]))

    ηv = ones(DAT, ncx+1, ncy+1)
    ηv[X.v.x.^2 .+ X.v.y.^2 .< rad^2] .= η_inc
  
    ηc = ones(DAT, ncx+2, ncy+2)
    ηc[X.c.x.^2 .+ X.c.y.^2 .< rad^2] .= η_inc
    # Arithmetic
    ηx = 0.5 .* (ηc[1:end-1,2:end-1] .+ ηc[2:end-0,2:end-1])
    ηy = 0.5 .* (ηc[2:end-1,1:end-1] .+ ηc[2:end-1,2:end-0])
    # Harmonic
    ηx = 2.0 ./ (1.0./ηc[1:end-1,2:end-1] .+ 1.0./ηc[2:end-0,2:end-1])
    ηy = 2.0 ./ (1.0./ηc[2:end-1,1:end-1] .+ 1.0./ηc[2:end-1,2:end-0])

    # # Direct viscosity evaluation
    # ηx = ones(Float32, ncx+1, ncy)
    # ηx[xv.x.^2 .+ (xc.y').^2 .< rad^2] .= 1e2
    # ηy = ones(Float32, ncx, ncy+1)
    # ηy[xc.x.^2 .+ (xv.y').^2 .< rad^2] .= 1e2

    # Maxloc business
    ηx2 = copy(ηx)
    maxloc!(ηx2, ηx)
    ηy2 = copy(ηy)
    maxloc!(ηy2, ηy)
    ηc2 = copy(ηc)
    maxloc!(ηc2, ηc)
    ηv2 = copy(ηv)
    maxloc!(ηv2, ηv)

    # Viscosity field
    η.x   .= to_device( ηx )
    η.y   .= to_device( ηy )
    PS     = 1
    V.x.c .= to_device( -ε̇bg*X.c.x*PS .+ ε̇bg*X.c.y*(1-PS) )
    V.x.v .= to_device( -ε̇bg*X.v.x*PS .+ ε̇bg*X.v.x*(1-PS) )
    V.y.c .= to_device( -  0*X.c.x    .+ ε̇bg*X.c.y*PS )
    V.y.v .= to_device( -  0*X.v.x    .+ ε̇bg*X.v.y*PS )

    # Time steps
    L      = sqrt( (xmax-xmin)^2 + (ymax-ymin)^2 )
    maxΔ   = max( maximum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), maximum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
    minΔ   = min( minimum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), minimum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
    # ΔτV.v .= to_device( cflV*max(Δ...)^2 ./ ηv2 )
    # ΔτV.c .= to_device( cflV*max(Δ...)^2 ./ ηc2 )
    # ΔτP.x .= to_device( cflP.*ηx.*min(Δ...)./L  )
    # ΔτP.y .= to_device( cflP.*ηy.*min(Δ...)./L  )
    ΔτV.v .= to_device( cflV*maxΔ^2 ./ ηv2)
    ΔτV.c .= to_device( cflV*maxΔ^2 ./ ηc2)
    ΔτP.x .= to_device( cflP.*ηx.*minΔ./L )
    ΔτP.y .= to_device( cflP.*ηy.*minΔ./L )

    kernel_StrainRates!    = StrainRates!(device)
    kernel_Stress!         = Stress!(device)
    kernel_Residuals!      = Residuals!(device)
    kernel_RateUpdate!     = RateUpdate!(device)
    kernel_SolutionUpdate! = SolutionUpdate!(device)
    TinyKernels.device_synchronize(get_device())

    for iter=1:itmax
        wait( kernel_StrainRates!(ε̇, ∇v, V, ∂, Δ; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Stress!(τ, η, ε̇; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Residuals!(R, τ, P, ∂, b, Δ; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP; ndrange=(ncx+2,ncy+2)) )
        if iter==1 || mod(iter,nout)==0
            errx = (; c = mean(abs.(R.x.c)), v = mean(abs.(R.x.v)) )
            erry = (; c = mean(abs.(R.y.c)), v = mean(abs.(R.y.v)) ) 
            errp = (; x = mean(abs.(∇v.x )), y = mean(abs.(∇v.y )) )
            if iter==1 errx0 = errx end
            @printf("Iteration %06d\n", iter)
            @printf("Rx = %2.2e %2.2e\n", errx.c, errx.v)
            @printf("Ry = %2.2e %2.2e\n", erry.c, erry.v)
            @printf("Rp = %2.2e %2.2e\n", errp.x, errp.y)
            if isnan(maximum(errx)) error("NaN à l'ail!") end
            if (maximum(errx)>1e3*maximum(errx0)) error("Stop before explosion!") end
            if maximum(errx) < ϵ && maximum(erry) < ϵ && maximum(errp) < ϵ
                break
            end
        end
    end

    #########################################################################################
    # Generate data
    vertx = [  X.v.x[1:end-1,1:end-1][:]  X.v.x[2:end-0,1:end-1][:]  X.v.x[2:end-0,2:end-0][:]  X.v.x[1:end-1,2:end-0][:] ] 
    verty = [  X.v.y[1:end-1,1:end-1][:]  X.v.y[2:end-0,1:end-1][:]  X.v.y[2:end-0,2:end-0][:]  X.v.y[1:end-1,2:end-0][:] ] 
    sol   = ( vx=to_host(V.x.c[2:end-1,2:end-1][:]), vy=to_host(V.y.c[2:end-1,2:end-1][:]), p=avWESN(to_host(P.x), to_host(P.y))[:], η=avWESN(to_host(η.x), to_host(η.y))[:])
    # xc2   = ∂η∂xvWESN(xc2_1, xc2_2[:,1:end-1])[:] 
    # yc2   = ∂η∂xvWESN(yc2_1, yc2_2[:,1:end-1])[:]
    PatchPlotMakie(vertx, verty, sol, minimum(X.v.x), maximum(X.v.x), minimum(X.v.y), maximum(X.v.y), write_fig=false)
     
    # Lx, Ly = xmax-xmin, ymax-ymin
    # f = Figure(resolution = ( Lx/Ly*600,600), fontsize=25, aspect = 2.0)

    # ax1 = Axis(f[1, 1], title = "P", xlabel = "x [km]", ylabel = "y [km]")
    # hm = heatmap!(ax1, xc.x, xv.y, to_host(P.y), colormap = (:turbo, 0.85), colorrange=(-3.,3.))
    # # hm = heatmap!(ax1, xc.x, xv.y, to_host(∇v.y), colormap = (:turbo, 0.85))
    # # hm = heatmap!(ax1, xv.x, xc.y, to_host(∇v.x), colormap = (:turbo, 0.85))
    # # hm = heatmap!(ax1, xce.x, xce.y, to_host(V.x.c), colormap = (:turbo, 0.85))
    # # hm = heatmap!(ax1, xce.x, xce.y, to_host(V.y.c), colormap = (:turbo, 0.85))
    # # hm = heatmap!(ax1, xv.x, xc.y, to_host(η.x), colormap = (:turbo, 0.85))

    # colsize!(f.layout, 1, Aspect(1, Lx/Ly))
    # GLMakie.Colorbar(f[1, 2], hm, label = "P", width = 20, labelsize = 25, ticklabelsize = 14 )
    # GLMakie.colgap!(f.layout, 20)
    # DataInspector(f)
    # display(f)

    # heatmap(xce.x, xce.y, to_host(V.x.c)')
    # heatmap(xce.x, xce.y, to_host(V.y.c)')
    # heatmap(xc.x, xv.y, to_host(∇v.y)')
    # heatmap(xv.x, xc.y, to_host(∇v.x)')
    # heatmap(xc.x, xv.y, to_host(ε̇.xy.y)')
    # heatmap(xv.x, xc.y, to_host(ε̇.xy.x)')
    # heatmap(xv.x, xc.y, to_host(η.x)')
    # heatmap(xc.x, xv.y, to_host(η.y)')
    # heatmap(xc.x, xv.y, to_host(τ.xx.y)')
    # heatmap(xv.x, xv.y, to_host(R.x.v)')
    # heatmap(xce.x, xce.y, to_host(R.x.c)')
    # heatmap(xv.x, xv.y, to_host(R.y.v)')

    # display(R.y.c)
    # display(R.x.c)
    # display(ε̇.xx.x)
    # display(ε̇.yy.x)

end

main(eletype; device)