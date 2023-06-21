# Surface processes following Simpson & Schlunegger (2003)
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

@tiny function Slopes!(∇h, h,  Δ)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(∇h.x.x) && j>1 && j<size(∇h.x.x,2) && i>1 && i<size(∇h.x.x,1) # BC
        dhdx = (h.c[i+1,j] - h.c[i,j]  )/Δ.x
        dhdy = (h.v[i,j]   - h.v[i,j-1])/Δ.y
        if abs(dhdx)<1e-17 dhdx = 1e-17 end
        if abs(dhdy)<1e-17 dhdy = 1e-17 end
        ∇h.x.x[i,j] = dhdx
        ∇h.y.x[i,j] = dhdy
    elseif i==1 || i==size(∇h.x.x,1)
        ∇h.x.x[i,j] = 1e-17
        ∇h.y.x[i,j] = 1e-17
    end
     if isin(∇h.x.y) && i>1 && i<size(∇h.x.y,1) && j<size(∇h.x.y,2) # BC
        dhdx = (h.v[i,j]   - h.v[i-1,j])/Δ.x
        dhdy = (h.c[i,j+1] - h.c[i,j])/Δ.y
        if abs(dhdx)<1e-17 dhdx = 1e-17 end
        if abs(dhdy)<1e-17 dhdy = 1e-17 end
        ∇h.x.y[i,j] = dhdx
        ∇h.y.y[i,j] = dhdy
     elseif j==size(∇h.x.y,2)
        ∇h.x.y[i,j] = 1e-17
        ∇h.y.y[i,j] = 1e-17
    end
end

@tiny function Fluxes!(q1, q2, D, ∇h, q, p)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(q1.x.x) && j>1 && j<size(∇h.x.x,2)
        D.x[i,j]    = (1.0 + p.De*q.x[i,j]^p.n)
        q1.x.x[i,j] = (1.0 + p.De*q.x[i,j]^p.n) * ∇h.x.x[i,j]
        q1.y.x[i,j] = (1.0 + p.De*q.x[i,j]^p.n) * ∇h.y.x[i,j]
        q2.x.x[i,j] = ∇h.x.x[i,j]/abs(∇h.x.x[i,j]) * q.x[i,j]
        q2.y.x[i,j] = ∇h.y.x[i,j]/abs(∇h.y.x[i,j]) * q.x[i,j]
    end
     if isin(q1.x.y) && i>1 && i<size(∇h.x.y,1)
        D.y[i,j]    = (1.0 + p.De*q.y[i,j]^p.n)
        q1.x.y[i,j] = (1.0 + p.De*q.y[i,j]^p.n) * ∇h.x.y[i,j]
        q1.y.y[i,j] = (1.0 + p.De*q.y[i,j]^p.n) * ∇h.y.y[i,j]
        q2.x.y[i,j] = ∇h.x.y[i,j]/abs(∇h.x.y[i,j]) * q.y[i,j]
        q2.y.y[i,j] = ∇h.y.y[i,j]/abs(∇h.y.y[i,j]) * q.y[i,j]
    end
end

@tiny function Residual_h!(R1, h, h0, q1, Δt, Δ)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(h.c) && i>1 && i<size(h.c,1) && j>1 && j<size(h.c,2)
        ∂q1dx = ( q1.x.x[i,j] - q1.x.x[i-1,j] )/Δ.x
        ∂q1dy = ( q1.y.y[i,j] - q1.y.y[i,j-1] )/Δ.y
        R1.c[i,j] = (h.c[i,j] -  h0.c[i,j])/Δt - (∂q1dx + ∂q1dy)
    end
    # if isin(h.v) && i>1 && i<size(h.v,1) && j>1 && j<size(h.v,2)
    if isin(h.v) && j>1 # BC
        ∂q1dx = ( q1.x.y[i+1,j] - q1.x.y[i,j] )/Δ.x
        ∂q1dy = ( q1.y.x[i,j+1] - q1.y.x[i,j] )/Δ.y
        R1.v[i,j] = (h.v[i,j] -  h0.v[i,j])/Δt - (∂q1dx + ∂q1dy)
    end
end

@tiny function Residual_q!(R2, q2, ∇h, Δ)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(R2.x) && i>1 && i<size(R2.x,1) && j>1 && j<size(R2.x,2)
        rhs = 0.
        rhs += (∇h.x.x[i,j]>0.) * (q2.x.x[i,j] - q2.x.x[i-1,j])/Δ.x + (∇h.x.x[i,j]<0.) * (q2.x.x[i+1,j] - q2.x.x[i,j])/Δ.x 
        rhs += (∇h.y.x[i,j]>0.) * (q2.y.x[i,j] - q2.y.x[i,j-1])/Δ.y + (∇h.y.x[i,j]<0.) * (q2.y.x[i,j+1] - q2.y.x[i,j])/Δ.y
        R2.x[i,j] = 1.0 + rhs 
    end
    if isin(R2.y) && i>1 && i<size(R2.y,1)&& j<size(R2.y,2) && j>1 
        rhs = 0.
        rhs += (∇h.x.y[i,j]>0.) * (q2.x.y[i,j] - q2.x.y[i-1,j])/Δ.x + (∇h.x.y[i,j]<0.) * (q2.x.y[i+1,j] - q2.x.y[i,j])/Δ.x
        rhs += (∇h.y.y[i,j]>0.) * (q2.y.y[i,j] - q2.y.y[i,j-1])/Δ.y + (∇h.y.y[i,j]<0.) * (q2.y.y[i,j+1] - q2.y.y[i,j])/Δ.y
        R2.y[i,j] = 1.0 + rhs
    end
end

@tiny function RateUpdate!(∂h∂τ, R1, θ)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(∂h∂τ.c) && i>1 && i<size(∂h∂τ.c,1) && j>1 && j<size(∂h∂τ.c,2)
        ∂h∂τ.c[i,j] = (1-θ) * ∂h∂τ.c[i,j] + R1.c[i,j]
    end
    # if isin(∂h∂τ.v) && i>1 && i<size(∂h∂τ.v,1) && j>1 && j<size(∂h∂τ.v,2)
    if isin(∂h∂τ.v) && j>1 # BC
        ∂h∂τ.v[i,j] = (1-θ) * ∂h∂τ.v[i,j] + R1.v[i,j]
    end
end

@tiny function Update_h!(h, ∂h∂τ, Δt1)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(h.c) && i>1 && i<size(h.c,1) && j>1 && j<size(h.c,2)
        h.c[i,j] -= Δt1*∂h∂τ.c[i,j]
    end
    # if isin(h.v) && i>1 && i<size(h.v,1) && j>1 && j<size(h.v,2)
    if isin(h.v) && j>1 # BC
        h.v[i,j] -= Δt1*∂h∂τ.v[i,j]
    end
end

@tiny function Update_q!(q, R2, Δt2)
    i, j = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j)
    if isin(q.x) && i>1 && i<size(q.x,1) && j>1 && j<size(q.x,2)
        q.x[i,j] -=  Δt2*R2.x[i,j]
    end
    if isin(q.y) && i>1 && i<size(q.y,1)  && j<size(q.y,2) #&& j>1
        q.y[i,j] -=  Δt2*R2.y[i,j]
    end
end


###############################

function main(::Type{DAT}; device) where DAT
    n          = 1
    # ncx, ncy   = n*120-2, n*120-2
    ncx, ncy   = 100, 100
    xmin, xmax = -0.0, 1.0
    ymin, ymax = -0.0, 1.0

    ϵ          = 1e-7     # nonlinear tolerence
    nt         = 200
    itmax      = 20000     # max number of iters
    nout       = 1000       # check frequency
    errx0      = (1. , 1.0)
    params     = (;De = 5e3, n = 3.0)
    θ          = 5/ncx
 

    q1    = VectorFSG2(DAT, device, :XY, ncx, ncy)
    q2    = VectorFSG2(DAT, device, :XY, ncx, ncy)
    ∇h    = VectorFSG2(DAT, device, :XY, ncx, ncy)
    D     = ScalarFSG(DAT, device, :XY, ncx, ncy)
    h     = ScalarFSG(DAT, device, :CV, ncx, ncy)
    h0    = ScalarFSG(DAT, device, :CV, ncx, ncy)
    q     = ScalarFSG(DAT, device, :XY, ncx, ncy)
    R1    = ScalarFSG(DAT, device, :CV, ncx, ncy)
    ∂h∂τ  = ScalarFSG(DAT, device, :CV, ncx, ncy)
    R2    = ScalarFSG(DAT, device, :XY, ncx, ncy)

    # Preprocessing
    Δ   = (; x =(xmax-xmin)/ncx, y =(ymax-ymin)/ncy)
    xce = (; x=LinRange(xmin-Δ.x/2, xmax+Δ.x/2, ncx+2), y=LinRange(ymin-Δ.y/2, ymax+Δ.y/2, ncy+2) ) 
    xv  = (; x=av1D(xce.x), y=av1D(xce.y) )
    xc  = (; x=av1D(xv.x),  y=av1D(xv.y) ) 

    xxv, yyv    = LinRange(xmin-Δ.x/2, xmax+Δ.x/2, 2ncx+3), LinRange(ymin-Δ.y/2, ymax+Δ.y/2, 2ncy+3)
    (xv4, yv4) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])
    X = (v=(x=xv4[2:2:end-1,2:2:end-1], y=yv4[2:2:end-1,2:2:end-1]), c=(x=xv4[1:2:end-0,1:2:end-0], y=yv4[1:2:end-0,1:2:end-0]), x=(x=xv4[2:2:end-1,3:2:end-2], y=yv4[2:2:end-1,3:2:end-2]), y=(x=xv4[3:2:end-2,2:2:end-1], y=yv4[3:2:end-2,2:2:end-1]))

    hc   = 0*xce.x .+ 0.1*xce.y'
    hv   =  zeros(ncx+1,ncy+1)
    hc .+= 0.01*rand(ncx+2, ncy+2)
    # hc .-= 0.01*exp.(- ((xce.x.-0.5).^2  .+ 0.0*xce.y')  ./ (0.1)^2 )
    hv .= 0.25.*(hc[1:end-1,1:end-1] .+ hc[2:end-0,1:end-1] .+ hc[1:end-1,2:end-0] .+ hc[2:end-0,2:end-0]) 
    hc[2:end-1,2:end-1] .= 0.25.*(hv[1:end-1,1:end-1] .+ hv[2:end-0,1:end-1] .+ hv[1:end-1,2:end-0] .+ hv[2:end-0,2:end-0]) 


    h.v .= to_device( hv )
    h.c .= to_device( hc )

    # # Time steps
    # L      = sqrt( (xmax-xmin)^2 + (ymax-ymin)^2 )
    # maxΔ   = max( maximum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), maximum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
    # minΔ   = min( minimum(X.v.x[2:end,:] .- X.v.x[1:end-1,:]), minimum(X.v.y[:,2:end] .- X.v.y[:,1:end-1])  )
    # ΔτV.v .= to_device( cflV*minΔ^2 ./ ηv2)
    # ΔτV.c .= to_device( cflV*minΔ^2 ./ ηc2)
    # ΔτP.x .= to_device( cflP.*ηx.*maxΔ./L )
    # ΔτP.y .= to_device( cflP.*ηy.*maxΔ./L )

    kernel_Slopes!     = Slopes!(device)
    kernel_Fluxes!     = Fluxes!(device)
    kernel_Residual_h! = Residual_h!(device)
    kernel_Residual_q! = Residual_q!(device)
    kernel_Update_h!   = Update_h!(device)
    kernel_Update_q!   = Update_q!(device)
    kernel_RateUpdate! = RateUpdate!(device)
    TinyKernels.device_synchronize(get_device())
    

    for it = 1:nt

        h0.v .= h.v 
        h0.c .= h.c 

        wait( kernel_Slopes!(∇h, h,  Δ; ndrange=(ncx+2,ncy+2)) )
        wait( kernel_Fluxes!(q1, q2, D, ∇h, q, params; ndrange=(ncx+2,ncy+2) ) )
        Δt1 = max(Δ...)^2 / 4.1 / maximum(to_host(D.x))  /params.De / 10
        Δt2 = max(Δ...)   / 4.1 / 1  # V is 1.0

        Δt = Δt1*5000

        # for iter=1:itmax
        #     wait( kernel_Slopes!(∇h, h,  Δ; ndrange=(ncx+2,ncy+2)) )
        #     wait( kernel_Fluxes!(q1, q2, D, ∇h, q, params; ndrange=(ncx+2,ncy+2) ) )
        #     wait( kernel_Residual_h!(R1, h, h0, q1, Δt, Δ; ndrange=(ncx+2,ncy+2) ) )
        #     wait( kernel_Residual_q!(R2, q2,  ∇h, Δ; ndrange=(ncx+2,ncy+2)) )
        #     wait( kernel_RateUpdate!(∂h∂τ, R1, θ; ndrange=(ncx+2,ncy+2)) )
        #     wait( kernel_Update_h!(h, ∂h∂τ, Δt1; ndrange=(ncx+2,ncy+2)) )
        #     wait( kernel_Update_q!(q, R2,   Δt2; ndrange=(ncx+2,ncy+2)) )
        #     if iter==1 || mod(iter,nout)==0
        #         err1 = (; c = mean(abs.(R1.c)), v = mean(abs.(R1.v)) )
        #         err2 = (; x = mean(abs.(R2.x)), y = mean(abs.(R2.y)) ) 
        #         if iter==1 err10 = err1 end
        #         @printf("it = %03d, iter = %05d, nR1=%1.6e nR2=%1.6e\n", it, iter, mean(err1), mean(err2))
        #         if isnan(maximum(err1)) error("NaN à l'ail!") end
        #         if (maximum(err1)>1e3) error("Stop before explosion!") end
        #         if maximum(err1) < ϵ && maximum(err2) < ϵ 
        #             break
        #         end
        #     end
        # end

        for iter=1:itmax
            wait( kernel_Slopes!(∇h, h,  Δ; ndrange=(ncx+2,ncy+2)) )
            wait( kernel_Fluxes!(q1, q2, D, ∇h, q, params; ndrange=(ncx+2,ncy+2) ) )
            wait( kernel_Residual_h!(R1, h, h0, q1, Δt, Δ; ndrange=(ncx+2,ncy+2) ) )
            wait( kernel_RateUpdate!(∂h∂τ, R1, θ; ndrange=(ncx+2,ncy+2)) )
            wait( kernel_Update_h!(h, ∂h∂τ, Δt1; ndrange=(ncx+2,ncy+2)) )
            if iter==1 || mod(iter,nout)==0
                err1 = (; c = mean(abs.(R1.c)), v = mean(abs.(R1.v)) )
                err2 = (; x = mean(abs.(R2.x)), y = mean(abs.(R2.y)) ) 
                if iter==1 err10 = err1 end
                @printf("it = %03d, iter = %05d, nR1=%1.6e nR2=%1.6e\n", it, iter, mean(err1), mean(err2))
                if isnan(maximum(err1)) error("NaN à l'ail!") end
                if (maximum(err1)>1e3) error("Stop before explosion!") end
                if maximum(err1) < ϵ && maximum(err2) < ϵ 
                    break
                end
            end
        end

        for iter=1:itmax
            wait( kernel_Slopes!(∇h, h,  Δ; ndrange=(ncx+2,ncy+2)) )
            wait( kernel_Fluxes!(q1, q2, D, ∇h, q, params; ndrange=(ncx+2,ncy+2) ) )
            wait( kernel_Residual_q!(R2, q2,  ∇h, Δ; ndrange=(ncx+2,ncy+2)) )
            wait( kernel_Update_q!(q, R2,   Δt2; ndrange=(ncx+2,ncy+2)) )
            if iter==1 || mod(iter,nout)==0
                err1 = (; c = mean(abs.(R1.c)), v = mean(abs.(R1.v)) )
                err2 = (; x = mean(abs.(R2.x)), y = mean(abs.(R2.y)) ) 
                if iter==1 err10 = err1 end
                @printf("it = %03d, iter = %05d, nR1=%1.6e nR2=%1.6e\n", it, iter, mean(err1), mean(err2))
                if isnan(maximum(err1)) error("NaN à l'ail!") end
                if (maximum(err1)>1e3) error("Stop before explosion!") end
                if maximum(err1) < ϵ && maximum(err2) < ϵ 
                    break
                end
            end
        end

        # @show h.c
        # @show ∇h.x.y
        # @show (q2.x.y)
        # @show (q2.y.y)
        # @show (R2.x)
        # @show (R2.y)

        # # #########################################################################################
        # # Generate data
        vertx = [  X.v.x[1:end-1,1:end-1][:]  X.v.x[2:end-0,1:end-1][:]  X.v.x[2:end-0,2:end-0][:]  X.v.x[1:end-1,2:end-0][:] ] 
        verty = [  X.v.y[1:end-1,1:end-1][:]  X.v.y[2:end-0,1:end-1][:]  X.v.y[2:end-0,2:end-0][:]  X.v.y[1:end-1,2:end-0][:] ] 
        # sol   = ( vx=to_host(V.x.c[2:end-1,2:end-1][:]), vy=to_host(V.y.c[2:end-1,2:end-1][:]), p=avWESN(to_host(P.x[:,2:end-1]), to_host(P.y[2:end-1,:]))[:], η=avWESN(to_host(η.x[:,2:end-1]), to_host(η.y[2:end-1,:]))[:])
        # # sol   = ( vx=to_host(R.x.c[2:end-1,2:end-1][:]), vy=to_host(R.y.c[2:end-1,2:end-1][:]), p=avWESN(to_host(P.x[:,2:end-1]), to_host(P.y[2:end-1,:]))[:], η=avWESN(to_host(η.x[:,2:end-1]), to_host(η.y[2:end-1,:]))[:])
        
        # Lx, Ly = xmax-xmin, ymax-ymin
        # f = Figure(resolution = ( Lx/Ly*1200,1200), fontsize=25, aspect = 2.0)

        # ax1 = Axis(f[1, 1], title = "q", xlabel = "x [km]", ylabel = "y [km]")
        # heatmap!(ax1, xv.x, xce.y, to_host(q.x), colormap = :turbo)

        # ax2 = Axis(f[1, 2], title = "q", xlabel = "x [km]", ylabel = "y [km]")
        # heatmap!(ax2, xce.x, xv.y, to_host(q.y), colormap = :turbo)

        # ax3 = Axis(f[2, 1], title = "h", xlabel = "x [km]", ylabel = "y [km]")
        # heatmap!(ax3, xce.x, xce.y, to_host(h.c), colormap = :turbo)

        # ax4 = Axis(f[2, 2], title = "h", xlabel = "x [km]", ylabel = "y [km]")
        # heatmap!(ax4, xv.x, xv.y, to_host(h.v), colormap = :turbo)

        # ga = f[3,1:2] = GridLayout()
        # ax5 = Axis3(ga[1,1])
        # surface!(ax5, xce.x[2:end-1], xce.y[2:end-1], to_host(h.c[2:end-1,2:end-1]), colormap=:turbo)

        # poly!(ax1, p, color = field, colormap = :turbo, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

        f = Figure(resolution = (1200,1200), fontsize=25, aspect = 2.0)

        ga = f[1,1] = GridLayout()
        ax1 = Axis3(ga[1,1])
        surface!(ax1, xce.x[2:end-1], xce.y[2:end-1], to_host(h.c[2:end-1,2:end-1]), colormap=:turbo)

        ga = f[2,1] = GridLayout()
        ax2 = Axis3(ga[1,1])
        surface!(ax2, xce.x[2:end-1], xce.y[2:end-1], to_host(h.c[2:end-1,2:end-1]), color=to_host(q.x[2:end-1,:]), colormap=:turbo)

        DataInspector(f)
        display(f)

    end
end

main(eletype; device)