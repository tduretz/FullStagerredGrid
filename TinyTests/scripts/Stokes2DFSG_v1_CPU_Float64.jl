# example triad 2D kernel
using TinyKernels, GLMakie, Printf
import LinearAlgebra: norm
import Statistics: mean
Makie.inline!(false)

@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end
@views av1D(x) = 0.5.*(x[2:end] .+ x[1:end-1])

# Introduce local time stepping
# PERFECT SCALING ON CPU AND FLOAT64!

include("DataStructures_v2.jl")
include("setup_example.jl")

# Select based upon your local device (:CPU, :CUDA, :AMDGPU, :Metal)
backend = :CPU

include("helpers.jl") # will be defined in TinyKernels soon

@setup_example()

###############################

@tiny function StrainRates!(ε̇, ∇v, V, Δ)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(∇v.x)
        ∂Vx∂x         = (V.x.c[ix+1,iy+1] - V.x.c[ix+0,iy+1])/Δ.x
        ∂Vy∂y         = (V.y.v[ix+0,iy+1] - V.y.v[ix+0,iy+0])/Δ.y
        ∂Vx∂y         = (V.x.v[ix+0,iy+1] - V.x.v[ix+0,iy+0])/Δ.y
        ∂Vy∂x         = (V.y.c[ix+1,iy+1] - V.y.c[ix+0,iy+1])/Δ.x
        ∇v.x[ix,iy]   = ∂Vx∂x + ∂Vy∂y
        ε̇.xx.x[ix,iy] = ∂Vx∂x - 1/3*(∂Vx∂x + ∂Vy∂y)
        ε̇.yy.x[ix,iy] = ∂Vy∂y - 1/3*(∂Vx∂x + ∂Vy∂y)
        ε̇.xy.x[ix,iy] = 1/2*(∂Vx∂y + ∂Vy∂x)
    end
    @inbounds if isin(∇v.y)
        ∂Vx∂x         = (V.x.v[ix+1,iy+0] - V.x.v[ix+0,iy+0])/Δ.x
        ∂Vy∂y         = (V.y.c[ix+1,iy+1] - V.y.c[ix+1,iy+0])/Δ.y
        ∂Vx∂y         = (V.x.c[ix+1,iy+1] - V.x.c[ix+1,iy+0])/Δ.y
        ∂Vy∂x         = (V.y.v[ix+1,iy+0] - V.y.v[ix+0,iy+0])/Δ.x
        ∇v.y[ix,iy]   = ∂Vx∂x + ∂Vy∂y
        ε̇.xx.y[ix,iy] = ∂Vx∂x - 1/3*(∂Vx∂x + ∂Vy∂y)
        ε̇.yy.y[ix,iy] = ∂Vy∂y - 1/3*(∂Vx∂x + ∂Vy∂y)
        ε̇.xy.y[ix,iy] = 1/2*(∂Vx∂y + ∂Vy∂x)
    end
end

@tiny function Stress!(τ, η, ε̇)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(τ.xx.x)
        τ.xx.x[ix,iy] = 2 * η.x[ix,iy] * ε̇.xx.x[ix,iy]
        τ.yy.x[ix,iy] = 2 * η.x[ix,iy] * ε̇.yy.x[ix,iy]
        τ.xy.x[ix,iy] = 2 * η.x[ix,iy] * ε̇.xy.x[ix,iy]
    end
    @inbounds if isin(τ.xx.y)
        τ.xx.y[ix,iy] = 2 * η.y[ix,iy] * ε̇.xx.y[ix,iy]
        τ.yy.y[ix,iy] = 2 * η.y[ix,iy] * ε̇.yy.y[ix,iy]
        τ.xy.y[ix,iy] = 2 * η.y[ix,iy] * ε̇.xy.y[ix,iy]
    end
end

@tiny function Residuals!(R, τ, P, b, Δ)
    ix, iy = @indices
    @inbounds if ix>1 && iy>1 && ix<size(R.x.v,1) && iy<size(R.x.v,2)
        R.x.v[ix,iy] = (τ.xx.y[ix,iy]-τ.xx.y[ix-1,iy])/Δ.x + (τ.xy.x[ix,iy]-τ.xy.x[ix,iy-1])/Δ.y - (P.y[ix,iy]-P.y[ix-1,iy])/Δ.x
        R.y.v[ix,iy] = (τ.yy.x[ix,iy]-τ.yy.x[ix,iy-1])/Δ.y + (τ.xy.y[ix,iy]-τ.xy.y[ix-1,iy])/Δ.x - (P.x[ix,iy]-P.x[ix,iy-1])/Δ.y
    end
    @inbounds if ix>1 && iy>1 && ix<size(R.x.c,1) && iy<size(R.x.c,2)
        R.x.c[ix,iy] = (τ.xx.x[ix,iy-1]-τ.xx.x[ix-1,iy-1])/Δ.x + (τ.xy.y[ix-1,iy]-τ.xy.y[ix-1,iy-1])/Δ.y - (P.x[ix,iy-1]-P.x[ix-1,iy-1])/Δ.x
        R.y.c[ix,iy] = (τ.yy.y[ix-1,iy]-τ.yy.y[ix-1,iy-1])/Δ.y + (τ.xy.x[ix,iy-1]-τ.xy.x[ix-1,iy-1])/Δ.x - (P.y[ix-1,iy]-P.y[ix-1,iy-1])/Δ.y
    end
end

@tiny function RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ)
    ix, iy = @indices
    @inbounds if ix>1 && iy>1 && ix<size(R.x.v,1) && iy<size(R.x.v,2)
        ∂V∂τ.x.v[ix,iy] = (1-θ) * ∂V∂τ.x.v[ix,iy] + R.x.v[ix,iy]
        ∂V∂τ.y.v[ix,iy] = (1-θ) * ∂V∂τ.y.v[ix,iy] + R.y.v[ix,iy]
    end 
    @inbounds if ix>1 && iy>1 && ix<size(R.x.c,1) && iy<size(R.x.c,2)
        ∂V∂τ.x.c[ix,iy] = (1-θ) * ∂V∂τ.x.c[ix,iy] + R.x.c[ix,iy]
        ∂V∂τ.y.c[ix,iy] = (1-θ) * ∂V∂τ.y.c[ix,iy] + R.y.c[ix,iy]
    end
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(∂P∂τ.x)
        ∂P∂τ.x[ix,iy] = -∇v.x[ix,iy]
    end
    @inbounds if isin(∂P∂τ.y)
        ∂P∂τ.y[ix,iy] = -∇v.y[ix,iy]
    end
end

@tiny function SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP)
    ix, iy = @indices
    @inbounds if ix>1 && iy>1 && ix<size(V.x.v,1) && iy<size(V.x.v,2)
        V.x.v[ix,iy] += ΔτV.v[ix,iy] * ∂V∂τ.x.v[ix,iy]
        V.y.v[ix,iy] += ΔτV.v[ix,iy] * ∂V∂τ.y.v[ix,iy]
    end 
    @inbounds if ix>1 && iy>1 && ix<size(V.x.c,1) && iy<size(V.x.c,2)
        V.x.c[ix,iy] += ΔτV.c[ix,iy] * ∂V∂τ.x.c[ix,iy]
        V.y.c[ix,iy] += ΔτV.c[ix,iy] * ∂V∂τ.y.c[ix,iy]
    end
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(P.x)
        P.x[ix,iy] += ΔτP.x[ix,iy] * ∂P∂τ.x[ix,iy]
    end
    @inbounds if isin(P.y)
        P.y[ix,iy] += ΔτP.y[ix,iy] * ∂P∂τ.y[ix,iy]
    end
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
    nout       = 500       # check frequency
    cflV       = 0.25
    cflP       = 1.0
    θ          = 3.0/ncx

    η    = ScalarFSG(DAT, device, :XY, ncx, ncy)
    P    = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ΔτP  = ScalarFSG(DAT, device, :XY, ncx, ncy)
    ΔτV  = ScalarFSG(DAT, device, :CV, ncx, ncy)
    ∇v   = ScalarFSG(DAT, device, :XY, ncx, ncy)
    τ    = TensorFSG(DAT, device, ncx, ncy)
    ε̇    = TensorFSG(DAT, device, ncx, ncy)
    V    = VectorFSG(DAT, device, ncx, ncy)
    R    = VectorFSG(DAT, device, ncx, ncy)
    b    = VectorFSG(DAT, device, ncx, ncy)
    ∂V∂τ = VectorFSG(DAT, device, ncx, ncy)
    ∂P∂τ = ScalarFSG(DAT, device, :XY, ncx, ncy)

    # Preprocessing
    Δ   = (; x =(xmax-xmin)/ncx, y =(ymax-ymin)/ncy)
    xce = (; x=LinRange(xmin-Δ.x/2, xmax+Δ.x/2, ncx+2), y=LinRange(ymin-Δ.y/2, ymax+Δ.y/2, ncy+2) ) 
    xv  = (; x=av1D(xce.x), y=av1D(xce.y) )
    xc  = (; x=av1D(xv.x),  y=av1D(xv.y) ) 

    ηv = ones(DAT, ncx+1, ncy+1)
    ηv[xv.x.^2 .+ (xv.y').^2 .< rad^2] .= 1e3
    # Arithmetic
    # ηx = 0.5 .* (ηv[:,1:end-1] .+ ηv[:,2:end-0]) # this gives cleanest inclusion pressure
    # ηy = 0.5 .* (ηv[1:end-1,:] .+ ηv[2:end-0,:])
    # Harmonic
    # ηx = 2.0 ./ (1.0./ηv[:,1:end-1] .+ 1.0./ηv[:,2:end-0])
    # ηy = 2.0 ./ (1.0./ηv[1:end-1,:] .+ 1.0./ηv[2:end-0,:])
    # # Geometric
    # ηx = sqrt.(ηv[:,1:end-1] .* ηv[:,2:end-0]) 
    # ηy = sqrt.(ηv[1:end-1,:] .* ηv[2:end-0,:])
  
    ηc = ones(DAT, ncx+2, ncy+2)
    ηc[xce.x.^2 .+ (xce.y').^2 .< rad^2] .= 1e3
    # Arithmetic
    ηx = 0.5 .* (ηc[1:end-1,2:end-1] .+ ηc[2:end-0,2:end-1])
    ηy = 0.5 .* (ηc[2:end-1,1:end-1] .+ ηc[2:end-1,2:end-0])
    # Harmonic
    ηx = 2.0 ./ (1.0./ηc[1:end-1,2:end-1] .+ 1.0./ηc[2:end-0,2:end-1])
    ηy = 2.0 ./ (1.0./ηc[2:end-1,1:end-1] .+ 1.0./ηc[2:end-1,2:end-0])
    # # Geometric
    # ηx = sqrt.(ηc[1:end-1,2:end-1] .* ηc[2:end-0,2:end-1])
    # ηy = sqrt.(ηc[2:end-1,1:end-1] .* ηc[2:end-1,2:end-0])

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
    V.x.c .= to_device( -ε̇bg*xce.x*PS .+ ε̇bg*xce.y'*(1-PS) )
    V.x.v .= to_device( -ε̇bg* xv.x*PS .+ ε̇bg* xv.y'*(1-PS) )
    V.y.c .= to_device( -  0*xce.x    .+ ε̇bg*xce.y'*PS )
    V.y.v .= to_device( -  0* xv.x    .+ ε̇bg* xv.y'*PS )

    # Time steps
    L      = sqrt( (xmax-xmin)^2 + (ymax-ymin)^2 )
    ΔτV.v .= to_device( cflV*max(Δ...)^2 ./ ηv2 )
    ΔτV.c .= to_device( cflV*max(Δ...)^2 ./ ηc2 )
    ΔτP.x .= to_device( cflP.*ηx.*min(Δ...)./L  )
    ΔτP.y .= to_device( cflP.*ηy.*min(Δ...)./L  )

    kernel_StrainRates!    = StrainRates!(device)
    kernel_Stress!         = Stress!(device)
    kernel_Residuals!      = Residuals!(device)
    kernel_RateUpdate!     = RateUpdate!(device)
    kernel_SolutionUpdate! = SolutionUpdate!(device)
    TinyKernels.device_synchronize(get_device())

    for iter=1:itmax
        kernel_StrainRates!(ε̇, ∇v, V, Δ; ndrange=(ncx+2,ncy+2))
        kernel_Stress!(τ, η, ε̇; ndrange=(ncx+2,ncy+2))
        kernel_Residuals!(R, τ, P, b, Δ; ndrange=(ncx+2,ncy+2))
        kernel_RateUpdate!(∂V∂τ, ∂P∂τ, R, ∇v, θ; ndrange=(ncx+2,ncy+2))
        kernel_SolutionUpdate!(V, P, ∂V∂τ, ∂P∂τ, ΔτV, ΔτP; ndrange=(ncx+2,ncy+2))
        if iter==1 || mod(iter,nout)==0
            errx = (; c = mean(abs.(R.x.c)), v = mean(abs.(R.x.v)) )
            erry = (; c = mean(abs.(R.y.c)), v = mean(abs.(R.y.v)) ) 
            errp = (; x = mean(abs.(∇v.x )), y = mean(abs.(∇v.y )) )
            @printf("Iteration %06d\n", iter)
            @printf("Rx = %2.2e %2.2e\n", errx.c, errx.v)
            @printf("Ry = %2.2e %2.2e\n", erry.c, erry.v)
            @printf("Rp = %2.2e %2.2e\n", errp.x, errp.y)
            if isnan(maximum(errx)) error("NaN à l'ail!") end
            if maximum(errx) < ϵ && maximum(erry) < ϵ && maximum(errp) < ϵ
                break
            end
        end
    end

    #########################################################################################
    Lx, Ly = xmax-xmin, ymax-ymin
    f = Figure(resolution = ( Lx/Ly*600,600), fontsize=25, aspect = 2.0)

    ax1 = Axis(f[1, 1], title = "P", xlabel = "x [km]", ylabel = "y [km]")
    hm = heatmap!(ax1, xc.x, xv.y, to_host(P.y), colormap = (:turbo, 0.85), colorrange=(-3.,3.))
    # hm = heatmap!(ax1, xc.x, xv.y, to_host(∇v.y), colormap = (:turbo, 0.85))
    # hm = heatmap!(ax1, xv.x, xc.y, to_host(∇v.x), colormap = (:turbo, 0.85))
    # hm = heatmap!(ax1, xce.x, xce.y, to_host(V.x.c), colormap = (:turbo, 0.85))
    # hm = heatmap!(ax1, xce.x, xce.y, to_host(V.y.c), colormap = (:turbo, 0.85))
    # hm = heatmap!(ax1, xv.x, xc.y, to_host(η.x), colormap = (:turbo, 0.85))

    colsize!(f.layout, 1, Aspect(1, Lx/Ly))
    GLMakie.Colorbar(f[1, 2], hm, label = "P", width = 20, labelsize = 25, ticklabelsize = 14 )
    GLMakie.colgap!(f.layout, 20)
    DataInspector(f)
    display(f)

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