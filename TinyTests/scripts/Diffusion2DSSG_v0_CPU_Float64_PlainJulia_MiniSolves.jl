# example triad 2D kernel
using  GLMakie, Printf
import LinearAlgebra: norm
import Statistics: mean
@views av1D(x) = 0.5.*(x[2:end] .+ x[1:end-1])
@views av2D_x(x) = 0.5.*(x[2:end,:] .+ x[1:end-1,:])
@views av2D_y(x) = 0.5.*(x[:,2:end] .+ x[:,1:end-1])
@views inn(x)    = x[2:end-1,2:end-1]
###############################

function main_PlainJulia(::Type{DAT}) where DAT
    n = 1
    ncx, ncy = n*120-2, n*120-2
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    ε̇bg      = -1
    rad      = 0.5
    ϵ        = 5e-8      # nonlinear tolerence
    iterMax  = 50000     # max number of iters
    nout     = 1000       # check frequency
    cfl      = 0.42

    # Allocate + initialise
    T     = zeros(DAT,  ncx+2, ncy+2); fill!(T, DAT(0.0))
    Ti    = zeros(DAT,  ncx+2, ncy+2); fill!(T, DAT(0.0))
    k     = zeros(DAT,  ncx+1, ncy+1); fill!(k, DAT(0.0))
    q     = (x=zeros(DAT,  ncx+1, ncy+0), y   = zeros(DAT,  ncx+0, ncy+1)); ; fill!(q.x, DAT(0.0)); ; fill!(q.y, DAT(0.0))
    b     = zeros(DAT,  ncx+0, ncy+0); fill!(b, DAT(0.0))
    RT    = zeros(DAT,  ncx+0, ncy+0); fill!(RT, DAT(0.0))
    Rθ    = zeros(DAT,  ncx+0, ncy+0); fill!(RT, DAT(0.0))
    ∂T∂τ  = zeros(DAT,  ncx+0, ncy+0); fill!(∂T∂τ, DAT(0.0))
    ∂T∂τ0 = zeros(DAT,  ncx+0, ncy+0); fill!(∂T∂τ, DAT(0.0))

    # Preprocessing
    Δ   = (; x =(xmax-xmin)/ncx, y =(ymax-ymin)/ncy)
    xce = (; x=LinRange(xmin-Δ.x/2, xmax+Δ.x/2, ncx+2), y=LinRange(ymin-Δ.y/2, ymax+Δ.y/2, ncy+2) ) 
    xv  = (; x=av1D(xce.x), y=av1D(xce.y) )
    xc  = (; x=av1D(xv.x),  y=av1D(xv.y) ) 

    # Model configuration
    k .= 1.
    b .= 1.
    k[(xv.x).^2 .+ (xv.y').^2 .< rad^2] .= 1.1
    b[(xc.x).^2 .+ (xc.y').^2 .< rad^2] .= 2.
    T .= -ε̇bg*xce.x .+ 0*xce.y'
    
    # PT params
    θ        = 4.3/ncx
    Δτ       = cfl*max(Δ...)^2 ./ maximum(k)  


    nl  = 30
    f   = zeros(nl)
    t   = zeros(nl)
    t   .= collect(LinRange(1e-3, 0.05, nl))
    θ0  = θ

    Ti .= T
    for i in eachindex(t)
        T    .= Ti
        ∂T∂τ .= 0.
        θ     = θ0#t[i]
        for iter=1:100
            ∂T∂τ0  .=  ∂T∂τ
            q.x    .= -av2D_y(k).*diff(T[:,2:end-1], dims=1)/Δ.x
            q.y    .= -av2D_x(k).*diff(T[2:end-1,:], dims=2)/Δ.y
            RT     .= b .- diff(q.x, dims=1)/Δ.x .- diff(q.y, dims=2)/Δ.y
            ∂T∂τ   .= (1.0-θ).*∂T∂τ0 .+ RT
            Rθ     .= ∂T∂τ .- (1.0-θ).*∂T∂τ0 .- RT
            inn(T) .+= Δτ.*∂T∂τ
            # if iter==1 || mod(iter,100)==0
            #     errT = mean(abs.(RT))
            #     errθ = mean(abs.(Rθ))
            #     @printf("It. %06d --- RT = %2.2e --- Rθ = %2.2e --- %2.2e %2.2e\n", iter, errT, errθ, θ0, θ)
            #     if isnan(errT) error("NaN à l'ail!") end
            #     if maximum(errT) < ϵ break end
            # end
        end
        f[i] = mean(abs.(RT))
    end 

    _, imin = findmin(f)




    # Rayleigh quotient
    
    # R = sqrt((q'*K*q)/(q'*M*q))
    RT2     = zeros(DAT,  ncx+2, ncy+2);
    Kr = zeros(DAT,  ncx+0, ncy+0);
    T2     = zeros(DAT,  ncx+2, ncy+2);
    inn(RT2) .= RT  
    q.x    .= -av2D_y(k).*diff(RT2[:,2:end-1], dims=1)/Δ.x
    q.y    .= -av2D_x(k).*diff(RT2[2:end-1,:], dims=2)/Δ.y
    Kr     .= .- diff(q.x, dims=1)/Δ.x .- diff(q.y, dims=2)/Δ.y 
    @show rKr = sqrt(-(RT[:]'*Kr[:]) / (RT[:]'* 1/Δτ *RT[:]))

    T    .= Ti
    ∂T∂τ .= 0.
#    @show θ     = t[imin]
    #  θ =  rKr
     θ =  θ0

    for iter=1:2000
        ∂T∂τ0  .=  ∂T∂τ
        q.x    .= -av2D_y(k).*diff(T[:,2:end-1], dims=1)/Δ.x
        q.y    .= -av2D_x(k).*diff(T[2:end-1,:], dims=2)/Δ.y
        RT     .= b .- diff(q.x, dims=1)/Δ.x .- diff(q.y, dims=2)/Δ.y
        ∂T∂τ   .= (1.0-θ).*∂T∂τ0 .+ RT
        Rθ     .= ∂T∂τ .- (1.0-θ).*∂T∂τ0 .- RT
        inn(T) .+= Δτ.*∂T∂τ
        if iter==1 || mod(iter,100)==0

           
            inn(RT2) .= RT  
            q.x    .= -av2D_y(k).*diff(RT2[:,2:end-1], dims=1)/Δ.x
            q.y    .= -av2D_x(k).*diff(RT2[2:end-1,:], dims=2)/Δ.y
            Kr     .= .- diff(q.x, dims=1)/Δ.x .- diff(q.y, dims=2)/Δ.y 
            θ1 = sqrt(abs(-(RT[:]'*Kr[:]) / (RT[:]'* 1/Δτ *RT[:])))
        
            q.x    .= -av2D_y(k).*diff(T[:,2:end-1], dims=1)/Δ.x
            q.y    .= -av2D_x(k).*diff(T[2:end-1,:], dims=2)/Δ.y
            Kr     .= .- diff(q.x, dims=1)/Δ.x .- diff(q.y, dims=2)/Δ.y 
            θ2 = 0.01+sqrt(abs(-(inn(T)[:]'*Kr[:]) / (inn(T)[:]'* 1/Δτ * inn(T)[:])))

            θ3 = 1/2*(θ1+θ2)
            # θ3 = θ2
            θ = θ3

            errT = mean(abs.(RT))
            errθ = mean(abs.(Rθ))
            @printf("It. %06d --- RT = %2.2e --- Rθ = %2.2e --- %2.2e %2.2e %2.2e\n", iter, errT, errθ, θ0, θ, θ3)
            if isnan(errT) error("NaN à l'ail!") end
            if maximum(errT) < ϵ break end
        end
    end

    @printf("%2.4e %2.4e\n", 4.3/ncx, t[imin],)

    Lx, Ly = xmax-xmin, ymax-ymin
    fig = Figure(resolution = ( Lx/Ly*600,600), fontsize=25, aspect = 2.0)
    ax1 = Axis(fig[1, 1], title = L"$k$", xlabel = "θ", ylabel = "f")
    # hm = heatmap!(ax1,  xc.x,  xv.y, (k), colormap = (:turbo, 0.85))
    # hm = heatmap!(ax1, xc.x, xc.y, (RT), colormap = (:turbo, 0.85))
    # hm = heatmap!(ax1, xc.x, xv.y, (q.y), colormap = (:turbo, 0.85))
    # hm = heatmap!(ax1, xce.x, xce.y, (T), colormap = (:turbo, 0.85))
    scatter!(ax1, t, f)
    # lines!(ax1, 4.3/ncx .*ones)
    # colsize!(fig.layout, 1, Aspect(1, Lx/Ly))
    # GLMakie.Colorbar(f[1, 2], hm, label = "τII", width = 20, labelsize = 25, ticklabelsize = 14 )
    # GLMakie.colgap!(fig.layout, 20)
    DataInspector(fig)
    display(fig)
end

main_PlainJulia(Float64)