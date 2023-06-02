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

# add ghost nodes for tensors and others
# add free surface
# add gravity
# add coordinate transformation
# ix --> i, iy -->j
# Introduce local time stepping
# PERFECT SCALING ON CPU AND FLOAT64!

include("DataStructures_v3.jl")
include("MeshDeformation.jl")
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

    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.p)]
    # poly!(p, color = sol.p, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # Axis(f[2,1], aspect = ar)
    # min_v = minimum( sol.vx ); max_v = maximum( sol.vx )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    # poly!(p, color = sol.vx, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # Axis(f[2,2], aspect = ar)
    min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    poly!(p, color = sol.vy, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    
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

@tiny function StrainRates!(ε̇, ∇v, V, P, η, ∂, Δ, options)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(ε̇.xx.x) && j>1 && j<size(ε̇.xx.x,2)
        ∂ξ∂x = ∂.ξ.∂x.x[i,j]; ∂ξ∂y = ∂.ξ.∂y.x[i,j]; ∂η∂x = ∂.η.∂x.x[i,j]; ∂η∂y = ∂.η.∂y.x[i,j];
        ∂Vx∂ξ       = ∂_∂ξ(V.x.c[i+1,j+1-1], V.x.c[i+0,j+1-1], Δ) 
        ∂Vy∂η       = ∂_∂η(V.y.v[i+0,j+1-1], V.y.v[i+0,j+0-1], Δ)
        ∂Vx∂η       = ∂_∂η(V.x.v[i+0,j+1-1], V.x.v[i+0,j+0-1], Δ)
        ∂Vy∂ξ       = ∂_∂ξ(V.y.c[i+1,j+1-1], V.y.c[i+0,j+1-1], Δ)
        ∂Vx∂x       = ∂Vx∂ξ * ∂ξ∂x + ∂Vx∂η * ∂η∂x
        ∂Vy∂y       = ∂Vy∂ξ * ∂ξ∂y + ∂Vy∂η * ∂η∂y
        ∂Vx∂y       = ∂Vx∂ξ * ∂ξ∂y + ∂Vx∂η * ∂η∂y
        ∂Vy∂x       = ∂Vy∂ξ * ∂ξ∂x + ∂Vy∂η * ∂η∂x
        ∇v.x[i,j] = ∂Vx∂x + ∂Vy∂y
        ε̇.xx.x[i,j] = ∂Vx∂x - 1//3*(∂Vx∂x + ∂Vy∂y)
        ε̇.yy.x[i,j] = ∂Vy∂y - 1//3*(∂Vx∂x + ∂Vy∂y)
        ε̇.xy.x[i,j] = 1//2*(∂Vx∂y + ∂Vy∂x)
    end
     if isin(ε̇.xx.y) && i>1 && i<size(ε̇.xx.y,1)
        ∂ξ∂x = ∂.ξ.∂x.y[i,j]; ∂ξ∂y = ∂.ξ.∂y.y[i,j]; ∂η∂x = ∂.η.∂x.y[i,j]; ∂η∂y = ∂.η.∂y.y[i,j];
        ∂Vx∂ξ       = ∂_∂ξ(V.x.v[i+1-1,j+0], V.x.v[i+0-1,j+0], Δ)
        ∂Vy∂η       = ∂_∂η(V.y.c[i+1-1,j+1], V.y.c[i+1-1,j+0], Δ)
        ∂Vx∂η       = ∂_∂η(V.x.c[i+1-1,j+1], V.x.c[i+1-1,j+0], Δ)
        ∂Vy∂ξ       = ∂_∂ξ(V.y.v[i+1-1,j+0], V.y.v[i+0-1,j+0], Δ)
        if options.free_surface && j == size(ε̇.xx.y, 2)
            ∂Vx∂η  = - ∂Vy∂ξ
            ∂Vy∂η  =   ∂Vx∂ξ + 0.75*P.y[i,j]/η.y[i,j] 
        end
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
    @inbounds if isin(τ.xx.x) && j>1 && j<size(τ.xx.x,2)
        τ.xx.x[i,j] = 2 * η.x[i,j] * ε̇.xx.x[i,j]
        τ.yy.x[i,j] = 2 * η.x[i,j] * ε̇.yy.x[i,j]
        τ.xy.x[i,j] = 2 * η.x[i,j] * ε̇.xy.x[i,j]
    end
    @inbounds if isin(τ.xx.y) && i>1 && i<size(τ.xx.y,1)
        τ.xx.y[i,j] = 2 * η.y[i,j] * ε̇.xx.y[i,j]
        τ.yy.y[i,j] = 2 * η.y[i,j] * ε̇.yy.y[i,j]
        τ.xy.y[i,j] = 2 * η.y[i,j] * ε̇.xy.y[i,j]
    end
end

@tiny function Residuals!(R, τ, P, ∂, b, Δ, options)
    i, j = @indices

    if options.free_surface North = size(R.x.v,2)+1
    else North = size(R.x.v,2) end

     if i>1 && j>1 && i<size(R.x.v,1) && j<North
        ∂ξ∂x = ∂.ξ.∂x.v[i,j]; ∂ξ∂y = ∂.ξ.∂y.v[i,j]; ∂η∂x = ∂.η.∂x.v[i,j]; ∂η∂y = ∂.η.∂y.v[i,j]
        
        ∂τxx∂ξ     = ∂_∂ξ(τ.xx.y[i+1,j], τ.xx.y[i-1+1,j], Δ)
        ∂τyy∂ξ     = ∂_∂ξ(τ.yy.y[i+1,j], τ.yy.y[i-1+1,j], Δ)
        ∂τxy∂ξ     = ∂_∂ξ(τ.xy.y[i+1,j], τ.xy.y[i-1+1,j], Δ)
        ∂P∂ξ       = ∂_∂ξ(   P.y[i+1,j],    P.y[i-1+1,j], Δ)

        ∂τxx∂η     = ∂_∂η(τ.xx.x[i,j+1], τ.xx.x[i,j-1+1], Δ)
        ∂τyy∂η     = ∂_∂η(τ.yy.x[i,j+1], τ.yy.x[i,j-1+1], Δ)
        ∂τyx∂η     = ∂_∂η(τ.xy.x[i,j+1], τ.xy.x[i,j-1+1], Δ)
        ∂P∂η       = ∂_∂η(   P.x[i,j+1],    P.x[i,j-1+1], Δ)
        
        ∂τxx∂x     = ∂τxx∂ξ * ∂ξ∂x + ∂τxx∂η * ∂η∂x
        ∂τyy∂y     = ∂τyy∂ξ * ∂ξ∂y + ∂τyy∂η * ∂η∂y 
        ∂τyx∂y     = ∂τxy∂ξ * ∂ξ∂y + ∂τyx∂η * ∂η∂y            # ⚠ assume τyx = τxy !!
        ∂τxy∂x     = ∂τxy∂ξ * ∂ξ∂x + ∂τyx∂η * ∂η∂x            # ⚠ assume τyx = τxy !!
        ∂P∂x       = ∂P∂ξ   * ∂ξ∂x + ∂P∂η   * ∂η∂x
        ∂P∂y       = ∂P∂ξ   * ∂ξ∂y + ∂P∂η   * ∂η∂y 
        R.x.v[i,j] = ∂τxx∂x + ∂τyx∂y - ∂P∂x
        R.y.v[i,j] = ∂τyy∂y + ∂τxy∂x - ∂P∂y + b.y.v[i,j]
    end
    @inbounds if i>1 && j>1 && i<size(R.x.c,1) && j<size(R.x.c,2)
        ∂ξ∂x = ∂.ξ.∂x.c[i,j]; ∂ξ∂y = ∂.ξ.∂y.c[i,j]; ∂η∂x = ∂.η.∂x.c[i,j]; ∂η∂y = ∂.η.∂y.c[i,j];

        ∂τxx∂ξ     = ∂_∂ξ(τ.xx.x[i,j-1+1], τ.xx.x[i-1,j-1+1], Δ)
        ∂τyy∂ξ     = ∂_∂ξ(τ.yy.x[i,j-1+1], τ.yy.x[i-1,j-1+1], Δ)
        ∂τxy∂ξ     = ∂_∂ξ(τ.xy.x[i,j-1+1], τ.xy.x[i-1,j-1+1], Δ)
        ∂P∂ξ       = ∂_∂ξ(   P.x[i,j-1+1],    P.x[i-1,j-1+1], Δ)

        ∂τxx∂η     = ∂_∂η(τ.xx.y[i-1+1,j], τ.xx.y[i-1+1,j-1], Δ)
        ∂τyy∂η     = ∂_∂η(τ.yy.y[i-1+1,j], τ.yy.y[i-1+1,j-1], Δ)
        ∂τyx∂η     = ∂_∂η(τ.xy.y[i-1+1,j], τ.xy.y[i-1+1,j-1], Δ)
        ∂P∂η       = ∂_∂η(   P.y[i-1+1,j],    P.y[i-1+1,j-1], Δ)

        ∂τxx∂x     = ∂τxx∂ξ * ∂ξ∂x + ∂τxx∂η * ∂η∂x
        ∂τyy∂y     = ∂τyy∂ξ * ∂ξ∂y + ∂τyy∂η * ∂η∂y 
        ∂τyx∂y     = ∂τxy∂ξ * ∂ξ∂y + ∂τyx∂η * ∂η∂y            # ⚠ assume τyx = τxy !!
        ∂τxy∂x     = ∂τxy∂ξ * ∂ξ∂x + ∂τyx∂η * ∂η∂x            # ⚠ assume τyx = τxy !!
        ∂P∂x       = ∂P∂ξ   * ∂ξ∂x + ∂P∂η   * ∂η∂x
        ∂P∂y       = ∂P∂ξ   * ∂ξ∂y + ∂P∂η   * ∂η∂y 
        R.x.c[i,j] = ∂τxx∂x + ∂τyx∂y - ∂P∂x  
        R.y.c[i,j] = ∂τyy∂y + ∂τxy∂x - ∂P∂y + b.y.c[i,j]
    end
end

@tiny function RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ, options)
    i, j = @indices

    if options.free_surface North = size(R.x.v,2)+1
    else North = size(R.x.v,2) end

    @inbounds if i>1 && j>1 && i<size(R.x.v,1) && j<North
        ∂V∂τ.x.v[i,j] = (1-θ) * ∂V∂τ.x.v[i,j] + R.x.v[i,j]
        ∂V∂τ.y.v[i,j] = (1-θ) * ∂V∂τ.y.v[i,j] + R.y.v[i,j]
    end 
    @inbounds if i>1 && j>1 && i<size(R.x.c,1) && j<size(R.x.c,2)
        ∂V∂τ.x.c[i,j] = (1-θ) * ∂V∂τ.x.c[i,j] + R.x.c[i,j]
        ∂V∂τ.y.c[i,j] = (1-θ) * ∂V∂τ.y.c[i,j] + R.y.c[i,j]
    end
    @inline isin(A) = checkbounds(Bool, A, i, j)
    @inbounds if isin(∂P∂τ.x) && j>1 && j<size(∂P∂τ.x,2)
        ∂P∂τ.x[i,j] = -∇v.x[i,j]
    end
    @inbounds if isin(∂P∂τ.y) && i>1 && i<size(∂P∂τ.y,1)
        ∂P∂τ.y[i,j] = -∇v.y[i,j]
    end
end

@tiny function SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP, options)
    i, j = @indices

    if options.free_surface North = size(V.x.v,2)+1
    else North = size(V.x.v,2) end

    @inbounds if i>1 && j>1 && i<size(V.x.v,1) && j<North
        V.x.v[i,j] += ΔτV.v[i,j] * ∂V∂τ.x.v[i,j]
        V.y.v[i,j] += ΔτV.v[i,j] * ∂V∂τ.y.v[i,j]
    end
    @inbounds if i>1 && j>1 && i<size(V.x.c,1) && j<size(V.x.c,2)
        V.x.c[i,j] += ΔτV.c[i,j] * ∂V∂τ.x.c[i,j]
        V.y.c[i,j] += ΔτV.c[i,j] * ∂V∂τ.y.c[i,j]
    end
    @inline isin(A) = checkbounds(Bool, A, i, j)
    @inbounds if isin(P.x) && j>1 && j<size(P.x,2)
        P.x[i,j] += ΔτP.x[i,j] * ∂P∂τ.x[i,j]
    end
    @inbounds if isin(P.y) && i>1 && i<size(P.y,1)
        P.y[i,j] += ΔτP.y[i,j] * ∂P∂τ.y[i,j]
    end
end

function Mesh_x( X, A, x0, σ, b, m, xmin0, xmax0, σx, options )
    xmin1 = (sinh.( σx.*(xmin0.-x0) ))
    xmax1 = (sinh.( σx.*(xmax0.-x0) ))
    sx    = (xmax0-xmin0)/(xmax1-xmin1)
    x     = (sinh.( σx.*(X[1].-x0) )) .* sx  .+ x0  
    # x = X[1]       
    return x
end

function Mesh_y( X, A, x0, σ, b, m, xmin0, xmax0, σx, options )
    xmin1 = (sinh.( σx.*(xmin0.-x0) ))
    xmax1 = (sinh.( σx.*(xmax0.-x0) ))
    sx    = (xmax0-xmin0)/(xmax1-xmin1)
    x     = (sinh.( σx.*(X[2].-x0) )) .* sx  .+ x0   
    # x = X[2]     
    return x
end

###############################

function main(::Type{DAT}; device) where DAT
    n          = 1
    ncx, ncy   = n*120-2, n*120-2
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    ε̇bg        = -1*0
    rad        = 0.5
    ϵ          = 5e-8      # nonlinear tolerence
    itmax      = 10000     # max number of iters
    errx0      = (1. , 1.0)
    nout       = 500       # check frequency
    θ          = 3.0/ncx
    η_inc      = 1.
    ρ_inc      = 2.
    g          = -1.0
    adapt_mesh = true
    # adapt_mesh = false
    options    = (; 
        free_surface = true,
    )

    if adapt_mesh
        cflV       = 0.25/4
        cflP       = 1.0*2
    else
        cflV       = 0.25
        cflP       = 1.0
    end
    # cflV       = 0.25/4
    # cflP       = 1.0/4

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
            xv4[i]   = Mesh_x( X_msh,  Amp, x0, σ, ymax, m, xmin, xmax, σx, options )
            yv4[i]   = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, options )
        end
        # # Compute forward transformation
        # params = (Amp=Amp, x0=x0, σ=σ, m=m, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, σx=σx, σy=σy, ϵ=ϵ)
        ∂x     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ∂y     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ComputeForwardTransformation_ini!( ∂x, ∂y, ξv, ηv, X_msh, Amp, x0, σ, m, xmin, xmax, ymin, ymax, σx, σy, ϵjac, options)
        # Solve for inverse transformation
        ∂ξ = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4))); ∂η = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4)))
        InverseJacobian!(∂ξ,∂η,∂x,∂y)
        # ∂ξ∂x .= ∂ξ.∂x; ∂ξ∂y .= ∂ξ.∂y
        # ∂η∂x .= ∂η.∂x; ∂η∂y .= ∂η.∂y
        ∂.ξ.∂x.x .= to_device( ∂ξ.∂x[2:2:end-1,1:2:end-0])
        ∂.ξ.∂x.y .= to_device( ∂ξ.∂x[1:2:end-0,2:2:end-1])
        ∂.ξ.∂x.c .= to_device( ∂ξ.∂x[1:2:end-0,1:2:end-0])
        ∂.ξ.∂x.v .= to_device( ∂ξ.∂x[2:2:end-1,2:2:end-1])
        ∂.ξ.∂y.x .= to_device( ∂ξ.∂y[2:2:end-1,1:2:end-0])
        ∂.ξ.∂y.y .= to_device( ∂ξ.∂y[1:2:end-0,2:2:end-1])
        ∂.ξ.∂y.c .= to_device( ∂ξ.∂y[1:2:end-0,1:2:end-0])
        ∂.ξ.∂y.v .= to_device( ∂ξ.∂y[2:2:end-1,2:2:end-1])
        ∂.η.∂x.x .= to_device( ∂η.∂x[2:2:end-1,1:2:end-0])
        ∂.η.∂x.y .= to_device( ∂η.∂x[1:2:end-0,2:2:end-1])
        ∂.η.∂x.c .= to_device( ∂η.∂x[1:2:end-0,1:2:end-0])
        ∂.η.∂x.v .= to_device( ∂η.∂x[2:2:end-1,2:2:end-1])
        ∂.η.∂y.x .= to_device( ∂η.∂y[2:2:end-1,1:2:end-0])
        ∂.η.∂y.y .= to_device( ∂η.∂y[1:2:end-0,2:2:end-1])
        ∂.η.∂y.c .= to_device( ∂η.∂y[1:2:end-0,1:2:end-0])
        ∂.η.∂y.v .= to_device( ∂η.∂y[2:2:end-1,2:2:end-1])
    else
        # Coordinate transformation
        ∂.ξ.∂x.x .= to_device( ones(ncx+1, ncy+2))
        ∂.ξ.∂x.y .= to_device( ones(ncx+2, ncy+1))
        ∂.ξ.∂x.c .= to_device( ones(ncx+2, ncy+2))
        ∂.ξ.∂x.v .= to_device( ones(ncx+1, ncy+1))
        ∂.η.∂y.x .= to_device( ones(ncx+1, ncy+2))
        ∂.η.∂y.y .= to_device( ones(ncx+2, ncy+1))
        ∂.η.∂y.c .= to_device( ones(ncx+2, ncy+2))
        ∂.η.∂y.v .= to_device( ones(ncx+1, ncy+1))
    end

    X = (v=(x=xv4[2:2:end-1,2:2:end-1], y=yv4[2:2:end-1,2:2:end-1]), c=(x=xv4[1:2:end-0,1:2:end-0], y=yv4[1:2:end-0,1:2:end-0]), x=(x=xv4[2:2:end-1,3:2:end-2], y=yv4[2:2:end-1,3:2:end-2]), y=(x=xv4[3:2:end-2,2:2:end-1], y=yv4[3:2:end-2,2:2:end-1]))
    
    ########### Buoyancy
    ρv = ones(DAT, ncx+1, ncy+1)
    ρv[X.v.x.^2 .+ X.v.y.^2 .< rad^2] .= ρ_inc

    ρc = ones(DAT, ncx+2, ncy+2)
    ρc[X.c.x.^2 .+ X.c.y.^2 .< rad^2] .= ρ_inc

    b.y.v   .= to_device( ρv*g )
    b.y.c   .= to_device( ρc*g )

    ########### Viscosity
    ηv = ones(DAT, ncx+1, ncy+1)
    ηv[X.v.x.^2 .+ X.v.y.^2 .< rad^2] .= η_inc
  
    ηc = ones(DAT, ncx+2, ncy+2)
    ηc[X.c.x.^2 .+ X.c.y.^2 .< rad^2] .= η_inc
    # Arithmetic
    ηx = 0.5 .* (ηc[1:end-1,:] .+ ηc[2:end-0,:])
    ηy = 0.5 .* (ηc[:,1:end-1] .+ ηc[:,2:end-0])
    # Harmonic
    ηx = 2.0 ./ (1.0./ηc[1:end-1,:] .+ 1.0./ηc[2:end-0,:])
    ηy = 2.0 ./ (1.0./ηc[:,1:end-1] .+ 1.0./ηc[:,2:end-0])

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
    @show size(ηx)
    @show size(ΔτP.x)
    ΔτP.x .= to_device( cflP.*ηx.*minΔ./L )
    ΔτP.y .= to_device( cflP.*ηy.*minΔ./L )

    kernel_StrainRates!    = StrainRates!(device)
    kernel_Stress!         = Stress!(device)
    kernel_Residuals!      = Residuals!(device)
    kernel_RateUpdate!     = RateUpdate!(device)
    kernel_SolutionUpdate! = SolutionUpdate!(device)
    TinyKernels.device_synchronize(get_device())

    for iter=1:itmax
        wait( kernel_StrainRates!(ε̇, ∇v, V, P, η, ∂, Δ, options; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Stress!(τ, η, ε̇; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Residuals!(R, τ, P, ∂, b, Δ, options; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ, options; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP, options; ndrange=(ncx+2,ncy+2)) )
        if iter==1 || mod(iter,nout)==0
            errx = (; c = mean(abs.(R.x.c)), v = mean(abs.(R.x.v)) )
            erry = (; c = mean(abs.(R.y.c)), v = mean(abs.(R.y.v)) ) 
            errp = (; x = mean(abs.(∇v.x )), y = mean(abs.(∇v.y )) )
            if iter==1 errx0 = errx end
            @printf("Iteration %06d\n", iter)
            @printf("Rx = %2.2e %2.2e\n", errx.c, errx.v)
            @printf("Ry = %2.2e %2.2e\n", erry.c, erry.v)
            @printf("Rp = %2.2e %2.2e\n", errp.x, errp.y)
            # if isnan(maximum(errx)) error("NaN à l'ail!") end
            # if (maximum(errx)>1e3*maximum(errx0)) error("Stop before explosion!") end
            if maximum(errx) < ϵ && maximum(erry) < ϵ && maximum(errp) < ϵ
                break
            end
        end
    end

    # #########################################################################################
    # Generate data
    vertx = [  X.v.x[1:end-1,1:end-1][:]  X.v.x[2:end-0,1:end-1][:]  X.v.x[2:end-0,2:end-0][:]  X.v.x[1:end-1,2:end-0][:] ] 
    verty = [  X.v.y[1:end-1,1:end-1][:]  X.v.y[2:end-0,1:end-1][:]  X.v.y[2:end-0,2:end-0][:]  X.v.y[1:end-1,2:end-0][:] ] 
    sol   = ( vx=to_host(V.x.c[2:end-1,2:end-1][:]), vy=to_host(V.y.c[2:end-1,2:end-1][:]), p=avWESN(to_host(P.x[:,2:end-1]), to_host(P.y[2:end-1,:]))[:], η=avWESN(to_host(η.x[:,2:end-1]), to_host(η.y[2:end-1,:]))[:])
    # PatchPlotMakie(vertx, verty, sol, minimum(X.v.x), maximum(X.v.x), minimum(X.v.y), maximum(X.v.y), write_fig=false)
    
    Lx, Ly = xmax-xmin, ymax-ymin
    f = Figure(resolution = ( Lx/Ly*1000,1000), fontsize=25, aspect = 2.0)

    ax1 = Axis(f[1, 1], title = "Surface Vy", xlabel = "x [km]", ylabel = "Vy [m/s]")
    lines!(ax1, X.v.x[2:end-1,end][:], to_host(V.y.v[2:end-1,end][:]))

    ax2 = Axis(f[1, 2], title = "Surface Vx", xlabel = "x [km]", ylabel = "Vx [m/s]")
    lines!(ax2, X.v.x[2:end-1,end][:], to_host(V.x.v[2:end-1,end][:]))

    ax2 = Axis(f[2, 1:2], title = "Vy", xlabel = "x [km]", ylabel = "y [km]")
    min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    poly!(ax2, p, color = sol.vy, colormap = :turbo, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    Colorbar(f[2, 3], colormap = :turbo, limits=limits, flipaxis = true, size = 25 )

    DataInspector(f)
    display(f)

end

main(eletype; device)