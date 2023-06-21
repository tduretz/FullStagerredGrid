# 3D Stokes taken from PT3D - https://github.com/tduretz/PT3D/blob/main/Stokes3D_Threads_v0.jl
using TinyKernels, Printf, GLMakie, Makie.GeometryBasics, MAT
import LinearAlgebra: norm
import Statistics: mean
Makie.inline!(false)

include("setup_example.jl")

# Select based upon your local device (:CPU, :CUDA, :AMDGPU, :Metal)
backend = :CPU

include("helpers.jl") # will be defined in TinyKernels soon

@setup_example()

###############################
   
function InitialCondition( Vx, Vy, Vz, ηv, ηc, ε, xv, yv, zv, xce, yce, zce, r )
    Threads.@threads for k = 1:length(zce)
        @inbounds for j = 1:length(yce)
            for i = 1:length(xce)
                if (i<=size(Vx,1)) Vx[i,j,k] = -ε*xv[i] end
                if (k<=size(Vz,3)) Vz[i,j,k] =  ε*zv[k] end
                if (i<=size(ηv,1) && j<=size(ηv,2) && k<=size(ηv,3)) 
                    ηv[i,j,k] = 1.0 
                    # if ( abs(zv[k]) < 0.2) ηv[i,j,k] = 1.1 end  
                    if ( (xv[i]*xv[i] + yv[j]*yv[j] + zv[k]*zv[k]) < r*r )  ηv[i,j,k] = 2.0 end  
                end
            end
        end
    end
    return nothing
end

###############################

@views function SetBoundaries( Vx, Vy, Vz )

    # West / East
    Vy[  1,:,:] .= Vy[  2,:,:]
    Vy[end,:,:] .= Vy[end-1,:,:]
    Vz[  1,:,:] .= Vz[  2,:,:]
    Vz[end,:,:] .= Vz[end-1,:,:]
    # Front / Back
    Vz[:,  1,:] .= Vz[:,  2,:]
    Vz[:,end,:] .= Vz[:,end-1,:]
    Vx[:,  1,:] .= Vx[:,  2,:]
    Vx[:,end,:] .= Vx[:,end-1,:]
    # South / North
    Vy[:,:,  1] .= Vy[:,:,  2]
    Vy[:,:,end] .= Vy[:,:,end-1]
    Vx[:,:,  1] .= Vx[:,:,  2]
    Vx[:,:,end] .= Vx[:,:,end-1]
    return nothing
end

###############################

function InterpV2C( ηc, ηv )
    Threads.@threads for k = 1:size(ηc,3)
        @inbounds for j = 1:size(ηc,2)
            for i = 1:size(ηc,1)
                ηc[i,j,k]  = 1.0/8.0*( ηv[i,  j,  k] + ηv[i+1,j,k  ] + ηv[i,j+1,k  ] + ηv[i,  j,k+1  ] )
                ηc[i,j,k] += 1.0/8.0*( ηv[i+1,j+1,k] + ηv[i+1,j,k+1] + ηv[i,j+1,k+1] + ηv[i+1,j+1,k+1] )
            end
        end
    end
    return nothing
end

###############################

function InterpV2xyz( ηxy, ηxz, ηyz, ηv )
    Threads.@threads for k = 1:size(ηv,3)
        @inbounds for j = 1:size(ηv,2)
            for i = 1:size(ηv,1)
                if (k<=size(ηxy,3)) ηxy[i,j,k]  = 1.0/2.0*( ηv[i,j,k] + ηv[i,j,k+1]) end
                if (j<=size(ηxz,2)) ηxz[i,j,k]  = 1.0/2.0*( ηv[i,j,k] + ηv[i,j+1,k]) end
                if (i<=size(ηyz,1)) ηyz[i,j,k]  = 1.0/2.0*( ηv[i,j,k] + ηv[i+1,j,k]) end
            end
        end
    end
    return nothing
end

###############################

function ComputeStrainRates( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Vx, Vy, Vz, Δx, Δy, Δz )
    _Δx, _Δy, _Δz = 1.0/Δx, 1.0/Δy, 1.0/Δz

    Threads.@threads for k = 1:size(εxx,3)
        @inbounds for j = 1:size(εxx,2)
            for i = 1:size(εxx,1)
                dVxΔx      = (Vx[i+1,j+1,k+1] - Vx[i,j+1,k+1]) * _Δx
                dVyΔy      = (Vy[i+1,j+1,k+1] - Vy[i+1,j,k+1]) * _Δy
                dVzΔz      = (Vz[i+1,j+1,k+1] - Vz[i+1,j+1,k]) * _Δz
                ∇V[i,j,k]  = dVxΔx + dVyΔy + dVzΔz
                εxx[i,j,k] = dVxΔx - 1//3 * ∇V[i,j,k]
                εyy[i,j,k] = dVyΔy - 1//3 * ∇V[i,j,k]
                εzz[i,j,k] = dVzΔz - 1//3 * ∇V[i,j,k]
            end
        end
    end

    Threads.@threads for k = 1:size(εxx,3)+1
        @inbounds for j = 1:size(εxx,2)+1
            for i = 1:size(εxx,1)+1
                if (i<=size(εxy,1)) && (j<=size(εxy,2)) && (k<=size(εxy,3))
                    dVxΔy      = (Vx[i,j+1,k+1] - Vx[i,j,k+1]) *_Δy 
                    dVyΔx      = (Vy[i+1,j,k+1] - Vy[i,j,k+1]) *_Δx 
                    εxy[i,j,k] = 1//2*(dVxΔy + dVyΔx)
                end
                if (i<=size(εxz,1)) && (j<=size(εxz,2)) && (k<=size(εxz,3))
                    dVxΔz      = (Vx[i  ,j+1,k+1] - Vx[i,j+1,k]) *_Δz                     
                    dVzΔx      = (Vz[i+1,j+1,k  ] - Vz[i,j+1,k]) *_Δx 
                    εxz[i,j,k] = 1//2*(dVxΔz + dVzΔx)
                end
                if (i<=size(εyz,1)) && (j<=size(εyz,2)) && (k<=size(εyz,3))
                    dVyΔz      = (Vy[i+1,j,k+1] - Vy[i+1,j,k]) *_Δz 
                    dVzΔy      = (Vz[i+1,j+1,k] - Vz[i+1,j,k]) *_Δy 
                    εyz[i,j,k] = 1//2*(dVyΔz + dVzΔy)
                end
            end
        end
    end
    return nothing
end

###############################

function ComputeStress( τxx, τyy, τzz, τxy, τxz, τyz, εxx, εyy, εzz, εxy, εxz, εyz, ηc, ηxy, ηxz, ηyz )

    Threads.@threads for k = 1:size(εxx,3)
        @inbounds for j = 1:size(εxx,2)
            for i = 1:size(εxx,1)
                τxx[i+1,j+1,k+1] = 2*ηc[i,j,k]*εxx[i,j,k]
                τyy[i+1,j+1,k+1] = 2*ηc[i,j,k]*εyy[i,j,k]
                τzz[i+1,j+1,k+1] = 2*ηc[i,j,k]*εzz[i,j,k]
            end
        end
    end

    Threads.@threads for k = 1:size(εxx,3)+1
        @inbounds for j = 1:size(εxx,2)+1
            for i = 1:size(εxx,1)+1
                if (i<=size(εxy,1)) && (j<=size(εxy,2)) && (k<=size(εxy,3))
                    τxy[i,j,k] = 2*ηxy[i,j,k]*εxy[i,j,k]
                end
                if (i<=size(εxz,1)) && (j<=size(εxz,2)) && (k<=size(εxz,3))
                    τxz[i,j,k] = 2*ηxz[i,j,k]*εxz[i,j,k]
                end
                if (i<=size(εyz,1)) && (j<=size(εyz,2)) && (k<=size(εyz,3))
                    τyz[i,j,k] = 2*ηyz[i,j,k]*εyz[i,j,k]
                end
            end
        end
    end
    return nothing
end

###############################

function ComputeResiduals( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, ∇V, Δx, Δy, Δz )

    _Δx, _Δy, _Δz = 1.0/Δx, 1.0/Δy, 1.0/Δz

    Threads.@threads for k = 1:size(P,3)-1
        @inbounds for j = 1:size(P,2)-1
            for i = 1:size(P,1)-1
                if (i<=size(Fx,1)) && (j<=size(Fx,2)) && (k<=size(Fx,3))
                    if (i>1 && i<size(Fx,1)) # avoid Dirichlets
                        Fx[i,j,k]  = (τxx[i+1,j+1,k+1] - τxx[i,j+1,k+1]) *_Δx
                        Fx[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i,j+1,k+1]) *_Δx
                        Fx[i,j,k] += (τxy[i,j+1,k] - τxy[i,j,k]) *_Δy
                        Fx[i,j,k] += (τxz[i,j,k+1] - τxz[i,j,k]) *_Δz
                    end
                end
                if (i<=size(Fy,1)) && (j<=size(Fy,2)) && (k<=size(Fy,3))
                    if (j>1 && j<size(Fy,2)) # avoid Dirichlets
                        Fy[i,j,k]  = (τyy[i+1,j+1,k+1] - τyy[i+1,j,k+1]) *_Δy
                        Fy[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i+1,j,k+1]) *_Δy
                        Fy[i,j,k] += (τxy[i+1,j,k] - τxy[i,j,k]) *_Δx
                        Fy[i,j,k] += (τyz[i,j,k+1] - τyz[i,j,k]) *_Δz
                    end
                end
                if (i<=size(Fz,1)) && (j<=size(Fz,2)) && (k<=size(Fz,3))
                    if (k>1 && k<size(Fz,3)) # avoid Dirichlets
                        Fz[i,j,k]  = (τzz[i+1,j+1,k+1] - τzz[i+1,j+1,k]) *_Δz
                        Fz[i,j,k] -= (  P[i+1,j+1,k+1] -   P[i+1,j+1,k]) *_Δz
                        Fz[i,j,k] += (τxz[i+1,j,k] - τxz[i,j,k]) *_Δx
                        Fz[i,j,k] += (τyz[i,j+1,k] - τyz[i,j,k]) *_Δy
                    end
                end
                if (i<=size(Fp,1)) && (j<=size(Fp,2)) && (k<=size(Fp,3))
                    Fp[i,j,k] = -∇V[i,j,k]
                end
            end
        end
    end
    return nothing
end

###############################

function UpdateRates( dVxdτ, dVydτ, dVzdτ, ρnum, Fx, Fy, Fz, ncx, ncy, ncz )

    Threads.@threads for k = 1:ncz+1
        @inbounds for j = 1:ncy+1
            for i = 1:ncx+1

                if (i<=size(Fx,1)) && (j<=size(Fx,2)) && (k<=size(Fx,3))
                    dVxdτ[i,j,k] = (1.0-ρnum)*dVxdτ[i,j,k] + Fx[i,j,k]
                end

                if (i<=size(Fy,1)) && (j<=size(Fy,2)) && (k<=size(Fy,3))
                    dVydτ[i,j,k] = (1.0-ρnum)*dVydτ[i,j,k] + Fy[i,j,k]
                end

                if (i<=size(Fz,1)) && (j<=size(Fz,2)) && (k<=size(Fz,3))
                    dVzdτ[i,j,k] = (1.0-ρnum)*dVzdτ[i,j,k] + Fz[i,j,k]
                end
            end
        end
    end

    return nothing
end

###############################

function UpdateVP( dVxdτ, dVydτ, dVzdτ, dPdτ, Vx, Vy, Vz, P, ρnum, Δτ, ΚΔτ,  ncx, ncy, ncz )
    Threads.@threads for k = 1:ncz+1
        @inbounds for j = 1:ncy+1
            for i = 1:ncx+1

                if (i<=size(dVxdτ,1)) && (j<=size(dVxdτ,2)) && (k<=size(dVxdτ,3))
                    Vx[i,j+1,k+1] += Δτ/ρnum*dVxdτ[i,j,k]
                end

                if (i<=size(dVydτ,1)) && (j<=size(dVydτ,2)) && (k<=size(dVydτ,3))
                    Vy[i+1,j,k+1] += Δτ/ρnum*dVydτ[i,j,k]
                end

                if (i<=size(dVzdτ,1)) && (j<=size(dVzdτ,2)) && (k<=size(dVzdτ,3))
                    Vz[i+1,j+1,k] += Δτ/ρnum*dVzdτ[i,j,k]
                end

                if (i<=size(dPdτ,1)) && (j<=size(dPdτ,2)) && (k<=size(dPdτ,3))
                    P[i+1,j+1,k+1] += ΚΔτ*dPdτ[i,j,k] 
                end
            end
        end
    end
    return nothing
end

#############################################################################################
#############################################################################################
#############################################################################################

function main( n )
    Lx,  Ly,  Lz  =  1.0,  1.0,  1.0 
    ncx, ncy, ncz = n*32, n*32, n*32
    BCtype = :PureShear_xz
    ε      = 1
    r      = 1e-1
    #-----------
    P   = zeros(ncx+2, ncy+2, ncz+2)
    Vx  = zeros(ncx+1, ncy+2, ncz+2)
    Vy  = zeros(ncx+2, ncy+1, ncz+2)
    Vz  = zeros(ncx+2, ncy+2, ncz+1)
    ηc  = zeros(ncx+0, ncy+0, ncz+0)
    ηv  = zeros(ncx+1, ncy+1, ncz+1)
    τxx = zeros(ncx+2, ncy+2, ncz+2)
    τyy = zeros(ncx+2, ncy+2, ncz+2)
    τzz = zeros(ncx+2, ncy+2, ncz+2)
    τxy = zeros(ncx+1, ncy+1, ncz+0)
    τxz = zeros(ncx+1, ncy+0, ncz+1)
    τyz = zeros(ncx+0, ncy+1, ncz+1)
    ηxy = zeros(ncx+1, ncy+1, ncz+0)
    ηxz = zeros(ncx+1, ncy+0, ncz+1)
    ηyz = zeros(ncx+0, ncy+1, ncz+1)
    ∇V  = zeros(ncx+0, ncy+0, ncz+0) 
    εxx = zeros(ncx+0, ncy+0, ncz+0) 
    εyy = zeros(ncx+0, ncy+0, ncz+0)
    εzz = zeros(ncx+0, ncy+0, ncz+0)
    εxy = zeros(ncx+1, ncy+1, ncz+0)
    εxz = zeros(ncx+1, ncy+0, ncz+1)
    εyz = zeros(ncx+0, ncy+1, ncz+1)
    Fx  = zeros(ncx+1, ncy+0, ncz+0)
    Fy  = zeros(ncx+0, ncy+1, ncz+0)
    Fz  = zeros(ncx+0, ncy+0, ncz+1)
    Fp  = zeros(ncx+0, ncy+0, ncz+0)
    dVxdτ = zeros(ncx+1, ncy+0, ncz+0)
    dVydτ = zeros(ncx+0, ncy+1, ncz+0)
    dVzdτ = zeros(ncx+0, ncy+0, ncz+1)
    #-----------
    xv = LinRange(-Lx/2, Lx/2, ncx+1)
    yv = LinRange(-Ly/2, Ly/2, ncy+1)
    zv = LinRange(-Lz/2, Lz/2, ncz+1)
    Δx, Δy, Δz = Lx/ncx, Ly/ncy, Lz/ncz
    xce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2)
    yce = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    zce = LinRange(-Lz/2-Δz/2, Lz/2+Δz/2, ncz+2)
    
    InitialCondition( Vx, Vy, Vz, ηv, ηc, ε, xv, yv, zv, xce, yce, zce, r )
    InterpV2C( ηc, ηv )
    InterpV2xyz( ηxy, ηxz, ηyz, ηv )
    
    niter  = 1e5
    nout   = 500
    Reopt  = 1*pi
    cfl    = 1.5
    ρnum   = cfl*Reopt/max(ncx,ncy,ncz)
    tol    = 1e-10
    Δτ     = ρnum*Δy^2 /maximum(ηc) /6.1 * cfl
    ΚΔτ    = cfl * maximum(ηc) * Δx / Lx 
    @printf("ρnum = %2.2e, Δτ = %2.2e, ΚΔτ = %2.2e %2.2e\n", ρnum, Δτ, ΚΔτ, maximum(ηc))
    
    for iter=1:niter
        SetBoundaries( Vx, Vy, Vz )
        ComputeStrainRates( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Vx, Vy, Vz, Δx, Δy, Δz )
        ComputeStress( τxx, τyy, τzz, τxy, τxz, τyz, εxx, εyy, εzz, εxy, εxz, εyz, ηc, ηxy, ηxz, ηyz )
        ComputeResiduals( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, ∇V, Δx, Δy, Δz )
        UpdateRates( dVxdτ, dVydτ, dVzdτ, ρnum, Fx, Fy, Fz, ncx, ncy, ncz )
        UpdateVP( dVxdτ, dVydτ, dVzdτ, Fp, Vx, Vy, Vz, P, ρnum, Δτ, ΚΔτ,  ncx, ncy, ncz )
        if mod(iter,nout) == 0 || iter==1
            nFx = norm(Fx)/sqrt(length(Fx))
            nFy = norm(Fy)/sqrt(length(Fy))
            nFz = norm(Fz)/sqrt(length(Fz))
            nFp = norm(Fp)/sqrt(length(Fp))
            @printf("Iter. %05d\n", iter) 
            @printf("Fx = %2.4e\n", nFx) 
            @printf("Fy = %2.4e\n", nFy) 
            @printf("Fz = %2.4e\n", nFz) 
            @printf("Fp = %2.4e\n", nFp) 
            max(nFx, nFy, nFz, nFp)<tol && break # short circuiting operations
            isnan(nFx) && error("NaN emergency!") 
            nFx>1e8    && error("Blow up!") 
        end
    end

    f = Figure(resolution = ( Lx/Ly*1200,1200), fontsize=25, aspect = 2.0)
    ax = Axis(f[1, 1], title = "Surface Vx", xlabel = "x [km]", ylabel = "Vx [m/s]")
    # heatmap!(ax, τxy[:, (size(τxy,2))÷2, :]')
    heatmap!(ax, to_host(εxx[:, size(τxy,2)÷2, :]))
    DataInspector(f)
    display(f)
    
    #-----------
    return nothing
end

#############################################################################################
#############################################################################################
#############################################################################################
 
    
@time main( 1 )
