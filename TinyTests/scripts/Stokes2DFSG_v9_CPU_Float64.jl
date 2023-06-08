# example triad 2D kernel
using TinyKernels, Printf, GLMakie, Makie.GeometryBasics, MAT
import LinearAlgebra: norm
import Statistics: mean
Makie.inline!(false)

@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end
@views av1D(x) = 0.5.*(x[2:end] .+ x[1:end-1])
@views avWESN(A,B) = 0.25*(A[1:end-1,:] .+ A[2:end-0,:] .+ B[:,1:end-1] .+ B[:,2:end-0])
@views ∂_∂ξ(aE, aW, Δ) = (aE - aW) / Δ.ξ
@views ∂_∂η(aN, aS, Δ) = (aN - aS) / Δ.η

# advect only surface, only vertically 
# add free surface stabilisation
# add basic explicit advection
# numerically computed forward transformation
# numerically computed slopes
# add topo
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

@tiny function StrainRates!(ε̇, ∇v, V, P, η, ∂, Δ, options, fsp)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(ε̇.xx.x) && j>1 && j<size(ε̇.xx.x,2) #&& i>1 && i<size(ε̇.xx.x,1) # FREE SLIP W/E
        ∂ξ∂x = ∂.ξ.∂x.x[i,j]; ∂ξ∂y = ∂.ξ.∂y.x[i,j]; ∂η∂x = ∂.η.∂x.x[i,j]; ∂η∂y = ∂.η.∂y.x[i,j];
        ∂Vx∂ξ       = ∂_∂ξ(V.x.c[i+1,j+1-1], V.x.c[i+0,j+1-1], Δ) 
        ∂Vy∂η       = ∂_∂η(V.y.v[i+0,j+1-1], V.y.v[i+0,j+0-1], Δ)
        ∂Vx∂η       = ∂_∂η(V.x.v[i+0,j+1-1], V.x.v[i+0,j+0-1], Δ)
        ∂Vy∂ξ       = ∂_∂ξ(V.y.c[i+1,j+1-1], V.y.c[i+0,j+1-1], Δ)
        ∂Vx∂x       = ∂Vx∂ξ * ∂ξ∂x + ∂Vx∂η * ∂η∂x
        ∂Vy∂y       = ∂Vy∂ξ * ∂ξ∂y + ∂Vy∂η * ∂η∂y
        ∂Vx∂y       = ∂Vx∂ξ * ∂ξ∂y + ∂Vx∂η * ∂η∂y
        ∂Vy∂x       = ∂Vy∂ξ * ∂ξ∂x + ∂Vy∂η * ∂η∂x
        ∇v.x[i,j]   = ∂Vx∂x + ∂Vy∂y
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
            ∂Vx∂η  = fsp.∂Vx∂∂Vx∂x[i] * ∂Vx∂ξ + fsp.∂Vx∂∂Vy∂x[i] * ∂Vy∂ξ +  fsp.∂Vx∂P[i] * P.y[i,j]
            ∂Vy∂η  = fsp.∂Vy∂∂Vx∂x[i] * ∂Vx∂ξ + fsp.∂Vy∂∂Vy∂x[i] * ∂Vy∂ξ +  fsp.∂Vy∂P[i] * P.y[i,j]/η.y[i,j] 
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

###############################

@tiny function Residuals!(R, τ, P, V, ∂, ρ, ∂ρ, g, θ, Δt, Δ, options)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if options.free_surface North = size(R.x.v,2)+1
    else North = size(R.x.v,2) end

     if isin(R.x.v) && j>1 && j<North #&& i>1 && i<size(R.x.v,1)
        ∂ξ∂x = ∂.ξ.∂x.v[i,j]; ∂ξ∂y = ∂.ξ.∂y.v[i,j]; ∂η∂x = ∂.η.∂x.v[i,j]; ∂η∂y = ∂.η.∂y.v[i,j]
        
        ∂τxx∂ξ     = ∂_∂ξ(τ.xx.y[i+1,j], τ.xx.y[i,j], Δ)
        ∂τyy∂ξ     = ∂_∂ξ(τ.yy.y[i+1,j], τ.yy.y[i,j], Δ)
        ∂τxy∂ξ     = ∂_∂ξ(τ.xy.y[i+1,j], τ.xy.y[i,j], Δ)
        ∂P∂ξ       = ∂_∂ξ(   P.y[i+1,j],    P.y[i,j], Δ)

        ∂τxx∂η     = ∂_∂η(τ.xx.x[i,j+1], τ.xx.x[i,j], Δ)
        ∂τyy∂η     = ∂_∂η(τ.yy.x[i,j+1], τ.yy.x[i,j], Δ)
        ∂τyx∂η     = ∂_∂η(τ.xy.x[i,j+1], τ.xy.x[i,j], Δ)
        ∂P∂η       = ∂_∂η(   P.x[i,j+1],    P.x[i,j], Δ)
        
        ∂τxx∂x     = ∂τxx∂ξ * ∂ξ∂x + ∂τxx∂η * ∂η∂x
        ∂τyy∂y     = ∂τyy∂ξ * ∂ξ∂y + ∂τyy∂η * ∂η∂y 
        ∂τyx∂y     = ∂τxy∂ξ * ∂ξ∂y + ∂τyx∂η * ∂η∂y            # ⚠ assume τyx = τxy !!
        ∂τxy∂x     = ∂τxy∂ξ * ∂ξ∂x + ∂τyx∂η * ∂η∂x            # ⚠ assume τyx = τxy !!
        ∂P∂x       = ∂P∂ξ   * ∂ξ∂x + ∂P∂η   * ∂η∂x
        ∂P∂y       = ∂P∂ξ   * ∂ξ∂y + ∂P∂η   * ∂η∂y 
        b          = -ρ.v[i,j]*g - Δt*θ*g*( ∂ρ.x.v[i,j]*V.x.v[i,j] + ∂ρ.y.v[i,j]*V.y.v[i,j] )
        if i>1  && i<size(R.x.v,1)
        R.x.v[i,j] = ∂τxx∂x + ∂τyx∂y - ∂P∂x
        else
            R.x.v[i,j] = 0.
        end
        R.y.v[i,j] = ∂τyy∂y + ∂τxy∂x - ∂P∂y - b
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
        b          = -ρ.c[i,j]*g - Δt*θ*g*( ∂ρ.x.c[i,j]*V.x.c[i,j] + ∂ρ.y.c[i,j]*V.y.c[i,j] )
        R.x.c[i,j] = ∂τxx∂x + ∂τyx∂y - ∂P∂x  
        R.y.c[i,j] = ∂τyy∂y + ∂τxy∂x - ∂P∂y - b
    end
end

###############################

@tiny function RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ, options)
    i, j = @indices

    if options.free_surface North = size(R.x.v,2)+1
    else North = size(R.x.v,2) end

    @inbounds if j>1  && j<North
        if i>1 && i<size(R.x.v,1) 
            ∂V∂τ.x.v[i,j] = (1-θ) * ∂V∂τ.x.v[i,j] + R.x.v[i,j] 
        end
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

###############################

@tiny function SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP, options)
    i, j = @indices

    if options.free_surface North = size(V.x.v,2)+1
    else North = size(V.x.v,2) end

    @inbounds if j>1 && j<North
        if i>1 && i<size(V.x.v,1)  
            V.x.v[i,j] += ΔτV.v[i,j] * ∂V∂τ.x.v[i,j] 
        end
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

###############################

function main(::Type{DAT}; device) where DAT
    n          = 1
    # ncx, ncy   = n*120-2, n*120-2
    ncx, ncy   = 81, 81
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -5.0, 0.0
    ε̇bg        = -1*0
    rad        = 0.5
    ϵ          = 1e-6      # nonlinear tolerence
    nt         = 1
    it0        = 0
    itmax      = 40000     # max number of iters
    nout       = 1000       # check frequency
    errx0      = (1. , 1.0)
    η_inc      = 1.
    ρ_inc      = 1.
    g          = -1.0
    adapt_mesh = true
    load_surface = false
    options      = (; 
        free_surface = true,
        swiss_x      = false,
        swiss_y      = false,
        topo         = true,
    )

    # PT params
    θ          = 1.0/ncx
    if adapt_mesh
        cflV       = 0.25/2
        cflP       = 1.0/2
    else
        cflV       = 0.25
        cflP       = 1.0
    end

    η    = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ∂ρ   = VectorFSG(DAT, device, ncx, ncy)
    ρ    = ScalarFSG(DAT, device, :CV, ncx, ncy)
    P    = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ΔτP  = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ΔτV  = ScalarFSG(DAT, device, :CV, ncx, ncy)
    ∇v   = ScalarFSG(DAT, device, :XY, ncx, ncy)
    τ    = TensorFSG(DAT, device, :XY, ncx, ncy)
    ε̇    = TensorFSG(DAT, device, :XY, ncx, ncy)
    V    = VectorFSG(DAT, device, ncx, ncy)
    R    = VectorFSG(DAT, device, ncx, ncy)
    ∂V∂τ = VectorFSG(DAT, device, ncx, ncy)
    ∂P∂τ = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ∂    = JacobianFSG(DAT, device, ncx, ncy)

    # Preprocessing
    Δ   = (; ξ =(xmax-xmin)/ncx, η =(ymax-ymin)/ncy)
    xce = (; x=LinRange(xmin-Δ.ξ/2, xmax+Δ.ξ/2, ncx+2), y=LinRange(ymin-Δ.η/2, ymax+Δ.η/2, ncy+2) ) 
    xv  = (; x=av1D(xce.x), y=av1D(xce.y) )
    xc  = (; x=av1D(xv.x),  y=av1D(xv.y) ) 
    
    xxv, yyv    = LinRange(xmin-Δ.ξ/2, xmax+Δ.ξ/2, 2ncx+3), LinRange(ymin-Δ.η/2, ymax+Δ.η/2, 2ncy+3)
    (xv4, yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])     
    X0        = (; ξ = copy(xv4), η = copy(yv4))

    # Initial topography
    x0     = (xmax + xmin)/2.
    m      = ymin
    Amp    = 1.0
    σ      = 0.5
    σx     = 0.5
    σy     = 0.5
    if load_surface  ################# RESTART
        it0  = 135
        file = matopen(string(@__DIR__, "/DeformedSurface0$(it0).mat"))
        h = read(file, "h") 
        close(file)
    else
        h      = Amp.*exp.(-(xv4.-x0).^2 ./ σ^2) 
    end

    if adapt_mesh
        # Deform mesh
        X_msh  = zeros(2)
        for i in eachindex(X0.ξ)          
            X_msh[1] = X0.ξ[i]
            X_msh[2] = X0.η[i] 
            xv4[i]   = Mesh_x( X_msh, h[i], x0, m, xmin, xmax, σx, options )
            yv4[i]   = Mesh_y( X_msh, h[i], x0, m, ymin, ymax, σy, options )
        end
        # Compute slope
        if options.topo 
            hx = -dhdx_num(xv4, yv4, Δ)
        end
        # # Compute forward transformation
        ∂x     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ∂y     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ComputeForwardTransformation!(∂x, ∂y, xv4, yv4, Δ)
        # Solve for inverse transformation
        ∂ξ = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4))); ∂η = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4)))
        InverseJacobian!(∂ξ,∂η,∂x,∂y)
        CopyJacobianToDevice!(∂, ∂ξ, ∂η)
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
    ρ_all    = ones(2ncx+3, 2ncx+3)
    ∂ρ∂ξ_x   = zeros(ncx+1, ncy+2)
    ∂ρ∂η_x   = zeros(ncx+1, ncy+2)
    ∂ρ∂ξ_y   = zeros(ncx+2, ncy+1)
    ∂ρ∂η_y   = zeros(ncx+2, ncy+1)
    ρ_all[xv4.^2 .+ yv4.^2 .< rad^2] .= ρ_inc
    ρ_all[:, end] .= 0. 
    ρ.v      .= to_device( ρ_all[2:2:end-1, 2:2:end-1] )
    ρ.c      .= to_device( ρ_all[1:2:end-0, 1:2:end-0] )
    ∂ρ∂ξ_x   .= (ρ.c[2:end,:] .- ρ.c[1:end-1,:])/Δ.ξ
    ∂ρ∂η_x[:,2:end-1] .= (ρ.v[:,2:end] .- ρ.v[:,1:end-1])/Δ.η
    ∂ρ∂ξ_y[2:end-1,:] .= (ρ.v[2:end,:] .- ρ.v[1:end-1,:])/Δ.ξ
    ∂ρ∂η_y   .= (ρ.c[:,2:end] .- ρ.c[:,1:end-1])/Δ.η
    ∂ρ∂x_x    = ∂ρ∂ξ_x .* ∂.ξ.∂x.x .+ ∂ρ∂η_x .* ∂.η.∂x.x
    ∂ρ∂x_y    = ∂ρ∂ξ_y .* ∂.ξ.∂x.y .+ ∂ρ∂η_y .* ∂.η.∂x.y
    ∂ρ∂y_x    = ∂ρ∂ξ_x .* ∂.ξ.∂y.x .+ ∂ρ∂η_x .* ∂.η.∂y.x
    ∂ρ∂y_y    = ∂ρ∂ξ_y .* ∂.ξ.∂y.y .+ ∂ρ∂η_y .* ∂.η.∂y.y
    ∂ρ∂x_c    = zeros(ncx+2, ncy+2)
    ∂ρ∂y_c    = zeros(ncx+2, ncy+2)
    ∂ρ∂x_c[2:end-1,2:end-1] .= avWESN(∂ρ∂x_x[:,2:end-1], ∂ρ∂x_y[2:end-1,:])
    ∂ρ∂x_v    = avWESN(∂ρ∂x_y, ∂ρ∂x_x)
    ∂ρ∂y_c[2:end-1,2:end-1] .= avWESN(∂ρ∂y_x[:,2:end-1], ∂ρ∂y_y[2:end-1,:])
    ∂ρ∂y_v    = avWESN(∂ρ∂y_y, ∂ρ∂y_x)
    ∂ρ.x.v   .= to_device( ∂ρ∂x_v )
    ∂ρ.x.c   .= to_device( ∂ρ∂x_c )
    ∂ρ.y.v   .= to_device( ∂ρ∂y_v )
    ∂ρ.y.c   .= to_device( ∂ρ∂y_c )

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

    # Initial topography
    xh0 = copy(xv4[:,end-1])
    yh0 = copy(yv4[:,end-1])

    # Topography
    free_surface_params = FreeSurfaceDiscretisation(ηy, ∂ξ, ∂η, hx )

    # Time steps
    Δt     = 0.1
    fss    = 1.0
    L      = sqrt( (xmax-xmin)^2 + (ymax-ymin)^2 )
    maxΔ   = max( maximum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), maximum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
    minΔ   = min( minimum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), minimum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
    ΔτV.v .= to_device( cflV*minΔ^2 ./ ηv2 /5)
    ΔτV.c .= to_device( cflV*minΔ^2 ./ ηc2 /5)
    ΔτP.x .= to_device( cflP.*ηx.*minΔ./L )
    ΔτP.y .= to_device( cflP.*ηy.*minΔ./L )

    kernel_StrainRates!    = StrainRates!(device)
    kernel_Stress!         = Stress!(device)
    kernel_Residuals!      = Residuals!(device)
    kernel_RateUpdate!     = RateUpdate!(device)
    kernel_SolutionUpdate! = SolutionUpdate!(device)
    TinyKernels.device_synchronize(get_device())

    # Time steps
    for it=it0+1:nt

        maxΔ   = max( maximum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), maximum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
        minΔ   = min( minimum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), minimum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
        ΔτV.v .= to_device( cflV*minΔ^2 ./ ηv2)
        ΔτV.c .= to_device( cflV*minΔ^2 ./ ηc2)
        ΔτP.x .= to_device( cflP.*ηx.*minΔ./L )
        ΔτP.y .= to_device( cflP.*ηy.*minΔ./L )

        # Iterations
        for iter=1:itmax
            wait( kernel_StrainRates!(ε̇, ∇v, V, P, η, ∂, Δ, options, free_surface_params; ndrange=(ncx+2,ncy+2)) )
            wait( kernel_Stress!(τ, η, ε̇; ndrange=(ncx+2,ncy+2)) )
            wait( kernel_Residuals!(R, τ, P, V, ∂, ρ, ∂ρ, g, fss, Δt, Δ, options; ndrange=(ncx+2,ncy+2)) )
            wait( kernel_RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ, options; ndrange=(ncx+2,ncy+2)) )
            wait( kernel_SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP, options; ndrange=(ncx+2,ncy+2)) )
            if iter==1 || mod(iter,nout)==0
                errx = (; c = mean(abs.(R.x.c)), v = mean(abs.(R.x.v)) )
                erry = (; c = mean(abs.(R.y.c)), v = mean(abs.(R.y.v)) ) 
                errp = (; x = mean(abs.(∇v.x )), y = mean(abs.(∇v.y )) )
                if iter==1 errx0 = errx end
                norm_Rx = 0.5*( norm(R.x.c[2:end-1,2:end-1])/sqrt(length(R.x.c[2:end-1,2:end-1])) + norm(R.x.v)/sqrt(length(R.x.v)) )
                norm_Ry = 0.5*( norm(R.y.c[2:end-1,2:end-1])/sqrt(length(R.y.c[2:end-1,2:end-1])) + norm(R.y.v)/sqrt(length(R.y.v)) )
                norm_Rp = 0.5*( norm(∇v.x[:,2:end-1])/sqrt(length(∇v.x[:,2:end-1])) + norm(∇v.y[2:end-1,:])/sqrt(length(∇v.y[2:end-1,:])) )
                @printf("it = %03d, iter = %05d, nRx=%1.6e nRy=%1.6e nRp=%1.6e\n", it, iter, norm_Rx, norm_Ry, norm_Rp)
                if isnan(maximum(errx)) error("NaN à l'ail!") end
                if (maximum(errx)>1e3) error("Stop before explosion!") end
                if maximum(errx) < ϵ && maximum(erry) < ϵ && maximum(errp) < ϵ
                    break
                end
            end
        end

        if adapt_mesh
            # Advect surface only vertically
            @show Δt  = min(Δ...)/maximum(abs.(V.y.c))/2.1/5
            Vy4 = zero(yv4)
            Vy4[2:2:end-1, 2:2:end-1] .= V.y.v 
            Vy4[1:2:end-0, 1:2:end-0] .= V.y.c
            Vy4[2:2:end-1, 3:2:end-2] .= 0.5*(V.y.v[:,2:end] .+ V.y.v[:,1:end-1])
            Vy4[3:2:end-2, 2:2:end-1] .= 0.5*(V.y.v[2:end,:] .+ V.y.v[1:end-1,:])
            h  .+= Δt.*Vy4[:,end-1]
            # Deform mesh
            X_msh  = zeros(2)
            for i in eachindex(X0.ξ)          
                X_msh[1] = X0.ξ[i]
                X_msh[2] = X0.η[i] 
                xv4[i]   = Mesh_x( X_msh, h[i], x0, m, xmin, xmax, σx, options )
                yv4[i]   = Mesh_y( X_msh, h[i], x0, m, ymin, ymax, σy, options )
            end
            # SAVE FOR RESTART !!!!!!!
            file = matopen(string(@__DIR__, "/DeformedSurface",@sprintf("%04d", it),".mat"), "w")
            write(file, "h", h)
            close(file)
            # Compute slope
            if options.topo hx = -dhdx_num(xv4, yv4, Δ) end
            # Solve for forward transformation
            ComputeForwardTransformation!(∂x, ∂y, xv4, yv4, Δ)        
            # Solve for inverse transformation
            ∂ξ = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4))); ∂η = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4)))
            InverseJacobian!(∂ξ,∂η,∂x,∂y)
            CopyJacobianToDevice!(∂, ∂ξ, ∂η)
            # Topography
            free_surface_params = FreeSurfaceDiscretisation(ηy, ∂ξ, ∂η, hx )
        end

        # #########################################################################################
        # Generate data
        X = (v=(x=xv4[2:2:end-1,2:2:end-1], y=yv4[2:2:end-1,2:2:end-1]), c=(x=xv4[1:2:end-0,1:2:end-0], y=yv4[1:2:end-0,1:2:end-0]), x=(x=xv4[2:2:end-1,3:2:end-2], y=yv4[2:2:end-1,3:2:end-2]), y=(x=xv4[3:2:end-2,2:2:end-1], y=yv4[3:2:end-2,2:2:end-1]))
        cell_vertx = [  X.v.x[1:end-1,1:end-1][:]  X.v.x[2:end-0,1:end-1][:]  X.v.x[2:end-0,2:end-0][:]  X.v.x[1:end-1,2:end-0][:] ] 
        cell_verty = [  X.v.y[1:end-1,1:end-1][:]  X.v.y[2:end-0,1:end-1][:]  X.v.y[2:end-0,2:end-0][:]  X.v.y[1:end-1,2:end-0][:] ] 
        node_vertx = [  X.c.x[1:end-1,1:end-1][:]  X.c.x[2:end-0,1:end-1][:]  X.c.x[2:end-0,2:end-0][:]  X.c.x[1:end-1,2:end-0][:] ] 
        node_verty = [  X.c.y[1:end-1,1:end-1][:]  X.c.y[2:end-0,1:end-1][:]  X.c.y[2:end-0,2:end-0][:]  X.c.y[1:end-1,2:end-0][:] ] 
        sol        = ( vx=to_host(V.x.c[2:end-1,2:end-1][:]), vy=to_host(V.y.c[2:end-1,2:end-1][:]), p=avWESN(to_host(P.x[:,2:end-1]), to_host(P.y[2:end-1,:]))[:], η=avWESN(to_host(η.x[:,2:end-1]), to_host(η.y[2:end-1,:]))[:])
        # sol   = ( vx=to_host(R.x.c[2:end-1,2:end-1][:]), vy=to_host(R.y.c[2:end-1,2:end-1][:]), p=avWESN(to_host(P.x[:,2:end-1]), to_host(P.y[2:end-1,:]))[:], η=avWESN(to_host(η.x[:,2:end-1]), to_host(η.y[2:end-1,:]))[:])
        
        Lx, Ly = xmax-xmin, ymax-ymin
        f = Figure(resolution = ( Lx/Ly*1200,1200), fontsize=25, aspect = 2.0)

        ax2 = Axis(f[1, 1], title = "Surface Vx", xlabel = "x [km]", ylabel = "Vx [m/s]")
        lines!(ax2, X.v.x[:,end][:], to_host(V.x.v[:,end][:]))
        # lines!(ax2, X.c.x[2:end-1,end-1][:], to_host(V.x.c[2:end-1,end-1][:]))

        ax1 = Axis(f[1, 3], title = "Surface Vy", xlabel = "x [km]", ylabel = "Vy [m/s]")
        lines!(ax1, X.v.x[:,end][:], to_host(V.y.v[:,end][:]))
        # lines!(ax1, X.c.x[2:end-1,end-1][:], to_host(V.y.c[2:end-1,end-1][:]))

        # ax2 = Axis(f[2, 1:2], title = "P", xlabel = "x [km]", ylabel = "y [km]")
        # min_v = minimum( sol.p ); max_v = maximum( sol.p )
        # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
        # p = [Polygon( Point2f0[ (cell_vertx[i,j], cell_verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
        # poly!(ax2, p, color = sol.p, colormap = :turbo, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
        # Colorbar(f[2, 3], colormap = :turbo, limits=limits, flipaxis = true, size = 25 )

        ax2 = Axis(f[2, 1], title = "Vx", xlabel = "x [km]", ylabel = "y [km]")
        min_v = minimum( sol.vx ); max_v = maximum( sol.vx )
        limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
        p = [Polygon( Point2f0[ (cell_vertx[i,j], cell_verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
        poly!(ax2, p, color = sol.vx, colormap = :turbo, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
        Colorbar(f[2, 2], colormap = :turbo, limits=limits, flipaxis = true, size = 25 )
        yh0 .= Amp.*exp.(-(xh0.-x0).^2 ./ σ^2) 
        lines!(ax2, xh0, yh0)

        ax2 = Axis(f[2, 3], title = "Vy", xlabel = "x [km]", ylabel = "y [km]")
        min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
        limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
        p = [Polygon( Point2f0[ (cell_vertx[i,j], cell_verty[i,j]) for j=1:4] ) for i in 1:length(sol.vy)]
        poly!(ax2, p, color = sol.vy, colormap = :turbo, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
        Colorbar(f[2, 4], colormap = :turbo, limits=limits, flipaxis = true, size = 25 )
        yh0 .= Amp.*exp.(-(xh0.-x0).^2 ./ σ^2) 
        lines!(ax2, xh0, yh0)

        # sol        = ( vx=to_host(V.x.v[:]), vy=to_host(V.y.v[:]), p=avWESN(to_host(P.x[:,2:end-1]), to_host(P.y[2:end-1,:]))[:], η=avWESN(to_host(η.x[:,2:end-1]), to_host(η.y[2:end-1,:]))[:])

        # ax2 = Axis(f[3, 1], title = "Vx", xlabel = "x [km]", ylabel = "y [km]")
        # min_v = minimum( sol.vx ); max_v = maximum( sol.vx )
        # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
        # p = [Polygon( Point2f0[ (node_vertx[i,j], node_verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
        # poly!(ax2, p, color = sol.vx, colormap = :turbo, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
        # Colorbar(f[3, 2], colormap = :turbo, limits=limits, flipaxis = true, size = 25 )
        # yh0 .= Amp.*exp.(-(xh0.-x0).^2 ./ σ^2) 
        # lines!(ax2, xh0, yh0)

        # ax2 = Axis(f[3, 3], title = "Vy", xlabel = "x [km]", ylabel = "y [km]")
        # min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
        # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
        # p = [Polygon( Point2f0[ (node_vertx[i,j], node_verty[i,j]) for j=1:4] ) for i in 1:length(sol.vy)]
        # poly!(ax2, p, color = sol.vy, colormap = :turbo, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
        # Colorbar(f[3, 4], colormap = :turbo, limits=limits, flipaxis = true, size = 25 )
        # yh0 .= Amp.*exp.(-(xh0.-x0).^2 ./ σ^2) 
        # lines!(ax2, xh0, yh0)

        # DataInspector(f)
        # save(string(@__DIR__, "/Fields"*@sprintf("%05d", it)*".png"), f) 

        display(f)
    end
end

main(eletype; device)