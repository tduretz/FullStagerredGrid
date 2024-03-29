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
            ∂Vx∂η  = fsp.∂Vx∂∂Vx∂x[i] * ∂Vx∂ξ + fsp.∂Vx∂∂Vy∂x[i] * ∂Vy∂ξ +  fsp.∂Vx∂P[i] * P.y[i,j]
            ∂Vy∂η  = fsp.∂Vy∂∂Vx∂x[i] * ∂Vx∂ξ + fsp.∂Vy∂∂Vy∂x[i] * ∂Vy∂ξ +  fsp.∂Vy∂P[i] * P.y[i,j]/η.y[i,j] 
             
            # println((fsp.∂Vx∂∂Vx∂x[i], fsp.∂Vx∂∂Vy∂x[i], fsp.∂Vx∂P[i]))
            # println((fsp.∂Vy∂∂Vx∂x[i], fsp.∂Vy∂∂Vy∂x[i], fsp.∂Vy∂P[i]))

            # ∂Vx∂η  = - ∂Vy∂ξ
            # ∂Vy∂η  =   0.5*∂Vx∂ξ + 0.75*P.y[i,j]/η.y[i,j] 
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
   if options.swiss_x
        xmin1 = (sinh.( σx.*(xmin0.-x0) ))
        xmax1 = (sinh.( σx.*(xmax0.-x0) ))
        sx    = (xmax0-xmin0)/(xmax1-xmin1)
        x     = (sinh.( σx.*(X[1].-x0) )) .* sx  .+ x0
    else
        x = X[1]
    end
    return x
end


function Mesh_y( X, A, y0, σ, b, m, ymin0, ymax0, σy, options )
    # y0    = ymax0
    y     = X[2]
    if options.swiss_y
        ymin1 = (sinh.( σy.*(ymin0.-y0) ))
        ymax1 = (sinh.( σy.*(ymax0.-y0) ))
        sy    = (ymax0-ymin0)/(ymax1-ymin1)
        y     = (sinh.( σy.*(X[2].-y0) )) .* sy  .+ y0
    end
    if options.topo
        z0    = -(A*exp(-(X[1]-y0)^2/σ^2) + b) # topography height
        y     = (y/ymin0)*((z0+m))-z0          # shift grid vertically
    end   
    return y
end

@views h(x,A,σ,b,x0)    = A*exp(-(x-x0)^2/σ^2) + b
@views dhdx(x,A,σ,b,x0) = -2*x/σ^2*A*exp(-(x-x0).^2/σ^2)

###############################

function main(::Type{DAT}; device) where DAT
    n          = 1
    # ncx, ncy   = n*120-2, n*120-2
    ncx, ncy   = 81, 81
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -5.0, 0.0
    ε̇bg        = -1*0
    rad        = 0.5
    ϵ          = 1e-10      # nonlinear tolerence
    it         = 1
    itmax      = 20000     # max number of iters
    nout       = 1000       # check frequency
    errx0      = (1. , 1.0)
    θ          = 3.0/ncx
    η_inc      = 1.
    ρ_inc      = 1.
    g          = -1.0
    adapt_mesh = true
    options    = (; 
        free_surface = true,
        swiss_x      = false,
        swiss_y      = false,
        topo         = true,
    )

    if adapt_mesh
        cflV       = 0.25/4
        cflP       = 1.0*2
    else
        cflV       = 0.25
        cflP       = 1.0
    end

    cflV       = 0.25  
    cflP       = 1.0

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
        hx     = zero(ξv)
        X_msh  = zeros(2)
        # Compute slope
        if options.topo hx = -dhdx.(ξv, Amp, σ, ymax, x0) end
        # Deform mesh
        for i in eachindex(ξv)          
            X_msh[1] = ξv[i]
            X_msh[2] = ηv[i]     
            xv4[i]   = Mesh_x( X_msh,  Amp, x0, σ, ymax, m, xmin, xmax, σx, options )
            yv4[i]   = Mesh_y( X_msh,  Amp, x0, σ, ymax, m, ymin, ymax, σy, options )
        end
        # # Compute forward transformation
        ∂x     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ∂y     = (∂ξ=zeros(size(yv4)), ∂η = zeros(size(yv4)) )
        ComputeForwardTransformation_ini!( ∂x, ∂y, ξv, ηv, X_msh, Amp, x0, σ, m, xmin, xmax, ymin, ymax, σx, σy, ϵjac, options)
        # Solve for inverse transformation
        ∂ξ = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4))); ∂η = (∂x=zeros(size(yv4)), ∂y=zeros(size(yv4)))
        InverseJacobian!(∂ξ,∂η,∂x,∂y)
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

    # @show ∂x.∂ξ
    # @show norm(ξv.-xv4)
    # @show norm(ηv.-yv4)

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

    ########### Topography # See python notebook v5
    η_surf   = ηy[:,end]
    dkdx     = ∂ξ.∂x[1:2:end, end-1]
    dkdy     = ∂ξ.∂y[1:2:end, end-1]
    dedx     = ∂η.∂x[1:2:end, end-1]
    dedy     = ∂η.∂y[1:2:end, end-1]
    h_x      =    hx[1:2:end, end]
    eta      = η_surf
    free_surface_params = (; # removed factor dz since we apply it directly to strain rates
        ∂Vx∂∂Vx∂x = (-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedy .* dkdx .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vx∂∂Vy∂x = (dedx .* dkdy .* h_x .^ 2 .+ 2 * dedx .* dkdy .- dedy .* dkdx .* h_x .^ 2 .- 2 * dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vx∂P     = (3 // 2) .* (dedx .* h_x .^ 2 .- dedx .+ 2 .* dedy .* h_x) ./ (eta .* (2 .* dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 .* dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 .* dedy .^ 2)),
        ∂Vy∂∂Vx∂x = (.-2 * dedx .* dkdy .* h_x .^ 2 .- dedx .* dkdy .+ 2 * dedy .* dkdx .* h_x .^ 2 .+ dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vy∂∂Vy∂x = (.-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedx .* dkdy .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vy∂P     = (3 // 2) .* (2 * dedx .* h_x .- dedy .* h_x .^ 2 .+ dedy) ./ (eta .* (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)),
    )
   
    # Time steps
    L      = sqrt( (xmax-xmin)^2 + (ymax-ymin)^2 )
    maxΔ   = max( maximum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), maximum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
    minΔ   = min( minimum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), minimum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
    ΔτV.v .= to_device( cflV*minΔ^2 ./ ηv2 /5)
    ΔτV.c .= to_device( cflV*minΔ^2 ./ ηc2 /5)
    ΔτP.x .= to_device( cflP.*ηx.*maxΔ./L )
    ΔτP.y .= to_device( cflP.*ηy.*maxΔ./L )

    # @show θ
    # @show cflV*minΔ^2 ./ ηv2[1]
    # @show cflV*minΔ^2 ./ ηc2[1]
    # @show  cflP.*ηx[1].*maxΔ./L 
    # @show  cflP.*ηy[1].*maxΔ./L 

    # θ = 0.0719948316447661
    # ΔτV.v .= to_device( 0.0019011669708533549.*ones(size(ηv)))
    # ΔτV.c .= to_device( 0.0019011669708533549.*ones(size(ηc)))
    # ΔτP.x .= to_device( 0.02619047619047619  .*ones(size(ηx)))
    # ΔτP.y .= to_device( 0.02619047619047619  .*ones(size(ηy)))


    kernel_StrainRates!    = StrainRates!(device)
    kernel_Stress!         = Stress!(device)
    kernel_Residuals!      = Residuals!(device)
    kernel_RateUpdate!     = RateUpdate!(device)
    kernel_SolutionUpdate! = SolutionUpdate!(device)
    TinyKernels.device_synchronize(get_device())

    for iter=1:itmax
        wait( kernel_StrainRates!(ε̇, ∇v, V, P, η, ∂, Δ, options, free_surface_params; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Stress!(τ, η, ε̇; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Residuals!(R, τ, P, ∂, b, Δ, options; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ, options; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP, options; ndrange=(ncx+2,ncy+2)) )
        if iter==1 || mod(iter,nout)==0
            errx = (; c = mean(abs.(R.x.c)), v = mean(abs.(R.x.v)) )
            erry = (; c = mean(abs.(R.y.c)), v = mean(abs.(R.y.v)) ) 
            errp = (; x = mean(abs.(∇v.x )), y = mean(abs.(∇v.y )) )
            if iter==1 errx0 = errx end
            # @printf("Iteration %06d\n", iter)
            # @printf("Rx = %2.2e %2.2e\n", errx.c, errx.v)
            # @printf("Ry = %2.2e %2.2e\n", erry.c, erry.v)
            # @printf("Rp = %2.2e %2.2e\n", errp.x, errp.y)
            norm_Rx = 0.5*( norm(R.x.c[2:end-1,2:end-1])/sqrt(length(R.x.c[2:end-1,2:end-1])) + norm(R.x.v)/sqrt(length(R.x.v)) )
            norm_Ry = 0.5*( norm(R.y.c[2:end-1,2:end-1])/sqrt(length(R.y.c[2:end-1,2:end-1])) + norm(R.y.v)/sqrt(length(R.y.v)) )
            norm_Rp = 0.5*( norm(∇v.x[:,2:end-1])/sqrt(length(∇v.x[:,2:end-1])) + norm(∇v.y[2:end-1,:])/sqrt(length(∇v.y[2:end-1,:])) )
            @printf("it = %03d, iter = %05d, nRx=%1.6e nRy=%1.6e nRp=%1.6e\n", it, iter, norm_Rx, norm_Ry, norm_Rp)
            # @printf("%1.6e %1.6e\n", norm(R.x.v)/sqrt(length(R.x.v)) , norm(R.x.c[2:end-1,2:end-1])/sqrt(length(R.x.c[2:end-1,2:end-1])))

            # norm_Rx = 0.5*( norm(V.x.c[2:end-1,2:end-1])/sqrt(length(V.x.c[2:end-1,2:end-1])) + norm(V.x.v)/sqrt(length(V.x.v)) )
            # norm_Ry = 0.5*( norm(V.y.c[2:end-1,2:end-1])/sqrt(length(V.y.c[2:end-1,2:end-1])) + norm(V.y.v)/sqrt(length(V.y.v)) )
            # norm_Rp = 0.5*( norm(P.x[:,2:end-1])/sqrt(length(P.x[:,2:end-1])) + norm(P.y[2:end-1,:])/sqrt(length(P.y[2:end-1,:])) )
            # @printf("it = %03d, iter = %05d, nRx=%1.6e nRy=%1.6e nRp=%1.6e\n", it, iter, norm_Rx, norm_Ry, norm_Rp)
            if isnan(maximum(errx)) error("NaN à l'ail!") end
            if (maximum(errx)>1e3) error("Stop before explosion!") end
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
    # sol   = ( vx=to_host(R.x.c[2:end-1,2:end-1][:]), vy=to_host(R.y.c[2:end-1,2:end-1][:]), p=avWESN(to_host(P.x[:,2:end-1]), to_host(P.y[2:end-1,:]))[:], η=avWESN(to_host(η.x[:,2:end-1]), to_host(η.y[2:end-1,:]))[:])
    
    Lx, Ly = xmax-xmin, ymax-ymin
    f = Figure(resolution = ( Lx/Ly*1000,1000), fontsize=25, aspect = 2.0)

    ax1 = Axis(f[1, 1], title = "Surface Vy", xlabel = "x [km]", ylabel = "Vy [m/s]")
    lines!(ax1, X.v.x[2:end-1,end][:], to_host(V.y.v[2:end-1,end][:]))

    ax2 = Axis(f[1, 2], title = "Surface Vx", xlabel = "x [km]", ylabel = "Vx [m/s]")
    lines!(ax2, X.v.x[2:end-1,end][:], to_host(V.x.v[2:end-1,end][:]))

    # ax2 = Axis(f[2, 1:2], title = "P", xlabel = "x [km]", ylabel = "y [km]")
    # min_v = minimum( sol.p ); max_v = maximum( sol.p )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    # poly!(ax2, p, color = sol.p, colormap = :turbo, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    # Colorbar(f[2, 3], colormap = :turbo, limits=limits, flipaxis = true, size = 25 )

    ax2 = Axis(f[2, 1:2], title = "Vy", xlabel = "x [km]", ylabel = "y [km]")
    min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    poly!(ax2, p, color = sol.vy, colormap = :turbo, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    Colorbar(f[2, 3], colormap = :turbo, limits=limits, flipaxis = true, size = 25 )

    DataInspector(f)
    display(f)
end

main(eletype; device)