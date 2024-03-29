# example triad 2D kernel
using TinyKernels, GLMakie, Printf
import LinearAlgebra: norm
import Statistics: mean
@views av1D(x) = 0.5.*(x[2:end] .+ x[1:end-1])

# PERFECT SCALING ON CPU AND FLOAT64!

include("setup_example.jl")

# Select based upon your local device (:CPU, :CUDA, :AMDGPU, :Metal)
backend = :CPU

include("helpers.jl") # will be defined in TinyKernels soon

@setup_example()

###############################

@tiny function Flux!(q, k, T, Δ)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
     if isin(q.x)
        kx         = 0.5*(k[ix,iy+1] + k[ix,iy]) 
        ∂T∂x       =  (T[ix+1,iy+1] - T[ix,iy+1]) / Δ.x
        q.x[ix,iy] = - kx * ∂T∂x
    end

     if isin(q.y)
        ky         = 0.5*(k[ix+1,iy] + k[ix,iy]) 
        ∂T∂y       =  (T[ix+1,iy+1] - T[ix+1,iy]) / Δ.y
        q.y[ix,iy] = - ky * ∂T∂y
    end
end

@tiny function Balance!(RT, Rθ, b, q, Δ)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
     if isin(RT)
        RT[ix,iy] = b[ix,iy] - (q.x[ix+1,iy] - q.x[ix,iy]) / Δ.x - (q.y[ix,iy+1] - q.y[ix,iy]) / Δ.y
    end
end

@tiny function Updates!(T, ∂T∂τ, RT, Rθ, Δτ, θ)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    if isin(∂T∂τ)
        ∂T∂τ0       = ∂T∂τ[ix,iy]
        ∂T∂τ[ix,iy] = (1-θ)* ∂T∂τ[ix,iy] + RT[ix,iy]
        Rθ[ix,iy]   = (∂T∂τ[ix,iy] - (1-θ)* ∂T∂τ0) - RT[ix,iy]
    end
    if ix>1 && iy>1 && ix<size(T,1) && iy<size(T,2)
        T[ix,iy]   += Δτ * ∂T∂τ[ix-1,iy-1]
    end
end

###############################

function main(::Type{DAT}; device) where DAT
    n = 1
    ncx, ncy = n*60-2, n*60-2
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    ε̇bg      = -1
    rad      = 0.5
    ϵ        = 5e-4      # nonlinear tolerence
    iterMax  = 50000     # max number of iters
    nout     = 1000       # check frequency
    cfl      = 0.42

    # Allocate + initialise
    T    = device_array(DAT, device, ncx+2, ncy+2); fill!(T, DAT(0.0))
    k    = device_array(DAT, device, ncx+1, ncy+1); fill!(k, DAT(0.0))
    q    = (x=device_array(DAT, device, ncx+1, ncy+0), y   = device_array(DAT, device, ncx+0, ncy+1)); ; fill!(q.x, DAT(0.0)); ; fill!(q.y, DAT(0.0))
    b    = device_array(DAT, device, ncx+0, ncy+0); fill!(b, DAT(0.0))
    RT   = device_array(DAT, device, ncx+0, ncy+0); fill!(RT, DAT(0.0))
    Rθ   = device_array(DAT, device, ncx+0, ncy+0); fill!(RT, DAT(0.0))
    ∂T∂τ = device_array(DAT, device, ncx+0, ncy+0); fill!(∂T∂τ, DAT(0.0))

    # Preprocessing
    Δ   = (; x =(xmax-xmin)/ncx, y =(ymax-ymin)/ncy)
    xce = (; x=LinRange(xmin-Δ.x/2, xmax+Δ.x/2, ncx+2), y=LinRange(ymin-Δ.y/2, ymax+Δ.y/2, ncy+2) ) 
    xv  = (; x=av1D(xce.x), y=av1D(xce.y) )
    xc  = (; x=av1D(xv.x),  y=av1D(xv.y) ) 

    # # Model configuration
    # k1  = ones(ncx+1, ncy+1)
    # k1[xv.x.^2 .+ (xv.y').^2 .< rad^2] .= 1.1  # shift of inclusion does not impact θ 
    # k  .= to_device(k1)
    # b1  = ones(ncx+0, ncy+0)
    # b1[xc.x.^2 .+ (xc.y').^2 .< rad^2] .= 2.
    # b  .= to_device(b1)
    # T1  = ones(ncx+2, ncy+2)
    # T1 .= -ε̇bg*xce.x .+ 0*xce.y'
    # T  .= to_device(T1)

    # # PT params
    # θ        = 4.3/ncx
    # Δτ       = cfl*max(Δ...)^2 ./ maximum(k) 

    # Model configuration
    k1  = ones(ncx+1, ncy+1)
    k1[(xv.x).^2 .+ (xv.y').^2 .< rad^2] .= 1.1
    k  .= to_device(k1)
    b1  = ones(ncx+0, ncy+0)
    b1[(xc.x.-3.).^2 .+ (xc.y'.-2.).^2 .< rad^2] .= 2.
    b  .= to_device(b1)
    T1  = ones(ncx+2, ncy+2)
    T1 .= -20*ε̇bg*xce.x .+ 20*xce.y'
    T  .= to_device(T1)
    
    # PT params
    θ = 5.3/ncx
    Δτ       = cfl*max(Δ...)^2 ./ maximum(k) 

    kernel_Flux!           = Flux!(device)
    kernel_Balance!        = Balance!(device)
    kernel_Updates!        = Updates!(device)
    TinyKernels.device_synchronize(get_device())

    for iter=1:10000
        wait( kernel_Flux!(q, k, T, Δ; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Balance!(RT, Rθ, b, q, Δ; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Updates!(T, ∂T∂τ, RT, Rθ, Δτ, θ; ndrange=(ncx+2,ncy+2)) )
        if iter==1 || mod(iter,100)==0
            errT = mean(abs.(RT))
            errθ = mean(abs.(Rθ))
            @printf("It. %06d --- RT = %2.2e --- Rθ = %2.2e\n", iter, errT, errθ)
            if isnan(errT) error("NaN à l'ail!") end
            if maximum(errT) < ϵ break end
        end
    end

    Lx, Ly = xmax-xmin, ymax-ymin
    f = Figure(resolution = ( Lx/Ly*600,600), fontsize=25, aspect = 2.0)
    ax1 = Axis(f[1, 1], title = L"$k$", xlabel = "x [km]", ylabel = "y [km]")
    hm = heatmap!(ax1,  xc.x,  xv.y, to_host(k), colormap = (:turbo, 0.85))
    # hm = heatmap!(ax1, xce.x, xce.y, to_host(θ), colormap = (:turbo, 0.85))
    hm = heatmap!(ax1, xc.x, xc.y, to_host(RT), colormap = (:turbo, 0.85))
    colsize!(f.layout, 1, Aspect(1, Lx/Ly))
    GLMakie.Colorbar(f[1, 2], hm, label = "τII", width = 20, labelsize = 25, ticklabelsize = 14 )
    GLMakie.colgap!(f.layout, 20)
    DataInspector(f)
    display(f)
end

main(eletype; device)