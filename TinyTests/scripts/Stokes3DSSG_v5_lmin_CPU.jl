# 3D Stokes taken from PT3D - https://github.com/tduretz/PT3D/blob/main/Stokes3D_Threads_v0.jl
# Use TinyKerkels instead of plain Julia
# DYREL
# Try VEP with center based formulation
using TinyKernels, Printf, WriteVTK, HDF5
using GLMakie, Makie.GeometryBasics, MAT
import LinearAlgebra: norm
import Statistics: mean
Makie.inline!(false)

include("setup_example.jl")

# Select based upon your local device (:CPU, :CUDA, :AMDGPU, :Metal)
backend = :CPU

include("helpers.jl") # will be defined in TinyKernels soon
include("Gershgorin3DSSG_centers.jl")

@setup_example()

av2x_arit(A) = 0.5.*(A[1:end-1,:,:] .+ A[2:end-0,:,:])   
av2y_arit(A) = 0.5.*(A[:,1:end-1,:] .+ A[:,2:end-0,:])
av2z_arit(A) = 0.5.*(A[:,:,1:end-1] .+ A[:,:,2:end-0])


av2xy_arit(A) = 0.25.*(A[1:end-1,1:end-1,:] .+ A[1:end-1,2:end-0,:] .+ A[2:end-0,1:end-1,:] .+ A[2:end-0,2:end-0,:])   
av2yz_arit(A) = 0.25.*(A[:,1:end-1,1:end-1] .+ A[:,1:end-1,2:end-0] .+ A[:,2:end-0,1:end-1] .+ A[:,2:end-0,2:end-0])
av2zx_arit(A) = 0.25.*(A[1:end-1,:,1:end-1] .+ A[1:end-1,:,2:end-0] .+ A[2:end-0,:,1:end-1] .+ A[2:end-0,:,2:end-0])

av2xy_harm(A) = 4.0./(1.0./A[1:end-1,1:end-1,:] .+ 1.0./A[1:end-1,2:end-0,:] .+ 1.0./A[2:end-0,1:end-1,:] .+ 1.0./A[2:end-0,2:end-0,:])   
av2yz_harm(A) = 4.0./(1.0./A[:,1:end-1,1:end-1] .+ 1.0./A[:,1:end-1,2:end-0] .+ 1.0./A[:,2:end-0,1:end-1] .+ 1.0./A[:,2:end-0,2:end-0])
av2zx_harm(A) = 4.0./(1.0./A[1:end-1,:,1:end-1] .+ 1.0./A[1:end-1,:,2:end-0] .+ 1.0./A[2:end-0,:,1:end-1] .+ 1.0./A[2:end-0,:,2:end-0])

av2xy_geom(A) = (A[1:end-1,1:end-1,:] .* A[1:end-1,2:end-0,:] .* A[2:end-0,1:end-1,:] .* A[2:end-0,2:end-0,:]).^(-4)   
av2yz_geom(A) = (A[:,1:end-1,1:end-1] .* A[:,1:end-1,2:end-0] .* A[:,2:end-0,1:end-1] .* A[:,2:end-0,2:end-0]).^(-4)
av2zx_geom(A) = (A[1:end-1,:,1:end-1] .* A[1:end-1,:,2:end-0] .* A[2:end-0,:,1:end-1] .* A[2:end-0,:,2:end-0]).^(-4)

###############################

@tiny function Kernel_MaxLoc!(A, B)
    i, j, k = @indices
    @inbounds if i>1 && j>1 && k>1 && i<size(A,1) && j<size(A,2) && k<size(A,3)
        A[i,j,k] = max( max( max( max(B[i-1,j  ,k  ], B[i+1,j  ,k  ])  , B[i  ,j  ,k  ] ),
        max(B[i  ,j-1,k  ], B[i  ,j+1,k  ]) ),
        max(B[i  ,j  ,k-1], B[i  ,j  ,k+1]) ) 
    end
end

###############################
   
@tiny function Kernel_InitialCondition!( Vx, Vy, Vz, ηv, ηc, Gv, Gc, ε, xv, yv, zv, xce, yce, zce, rc )
    i, j, k = @indices
    @inbounds if i<length(xce) && k<length(zce) && j<length(yce)
        if (i<=size(Vx,1)) Vx[i,j,k] = -ε*xv[i] end
        if (k<=size(Vz,3)) Vz[i,j,k] =  ε*zv[k] end
        if (i<=size(ηv,1) && j<=size(ηv,2) && k<=size(ηv,3)) 
            ηv[i,j,k] = 1.0e10 
            Gv[i,j,k] = 1.0
            x = xv[i] - (-0.5)
            y = yv[j]
            z = zv[k] - (-0.8150/2) 
            r = rc
            if ( (x*x + 0*y*y + z*z) < r*r )  Gv[i,j,k] = 0.25 end  

            x = xv[i] - (0.5)
            y = yv[j]
            z = zv[k] - (-0.8150/4) 
            if ( (x*x + 0*y*y + z*z) < r*r )  Gv[i,j,k] = 0.25 end

            x = xv[i] 
            y = yv[j]
            z = zv[k] - 0*(+0.8150) 
            if ( (x*x + 0*y*y + z*z) < r*r )  Gv[i,j,k] = 0.25 end
        end
    end
    return
end

###############################

@tiny function Kernel_SetBoundaries!( Vx, Vy, Vz )
    i, j, k = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j, k)
    # Vx
    @inbounds if isin(Vx)
        # Y: Front / Back
        if j==1          Vx[i,j,k] = Vx[i,j+1,k] end
        if j==size(Vx,2) Vx[i,j,k] = Vx[i,j-1,k] end
        # Z: South / North
        if k==1          Vx[i,j,k] = Vx[i,j,k+1] end
        if k==size(Vx,3) Vx[i,j,k] = Vx[i,j,k-1] end
    end
    # Vy
    @inbounds if isin(Vy)
        # X: West / East
        if i==1          Vy[i,j,k] = Vy[i+1,j,k] end
        if i==size(Vy,1) Vy[i,j,k] = Vy[i-1,j,k] end
        # Z: South / North
        if k==1          Vy[i,j,k] = Vy[i,j,k+1] end
        if k==size(Vy,3) Vy[i,j,k] = Vy[i,j,k-1] end
    end
    # Vz
    @inbounds if isin(Vz)
        # X: West / East
        if i==1          Vz[i,j,k] = Vz[i+1,j,k] end
        if i==size(Vz,1) Vz[i,j,k] = Vz[i-1,j,k] end
        # Y: Front / Back
        if j==1          Vz[i,j,k] = Vz[i,j+1,k] end
        if j==size(Vz,2) Vz[i,j,k] = Vz[i,j-1,k] end
    end
    return
end

# ###############################

@tiny function Kernel_InterpV2C!( ηc, ηv )
    i, j, k = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j, k)
    @inbounds if isin(ηc)
        ηc[i,j,k]  = 1.0/8.0*( ηv[i,  j,  k] + ηv[i+1,j,k  ] + ηv[i,j+1,k  ] + ηv[i,  j,k+1  ] )
        ηc[i,j,k] += 1.0/8.0*( ηv[i+1,j+1,k] + ηv[i+1,j,k+1] + ηv[i,j+1,k+1] + ηv[i+1,j+1,k+1] )
    end
    return
end

# ###############################

@tiny function Kernel_ComputeStrainRates!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Vx, Vy, Vz, Δx, Δy, Δz )
    _Δx, _Δy, _Δz = 1.0/Δx, 1.0/Δy, 1.0/Δz
    i, j, k = @indices
    @inbounds if (i<=size(εxx,1)+0) && (j<=size(εxx,2)+0) && (k<=size(εxx,3)+0)
        dVxΔx      = (Vx[i+1,j+1,k+1] - Vx[i,j+1,k+1]) * _Δx
        dVyΔy      = (Vy[i+1,j+1,k+1] - Vy[i+1,j,k+1]) * _Δy
        dVzΔz      = (Vz[i+1,j+1,k+1] - Vz[i+1,j+1,k]) * _Δz
        ∇V[i,j,k]  = dVxΔx + dVyΔy + dVzΔz
        εxx[i,j,k] = dVxΔx - 1.0/3.0 * ∇V[i,j,k]
        εyy[i,j,k] = dVyΔy - 1.0/3.0 * ∇V[i,j,k]
        εzz[i,j,k] = dVzΔz - 1.0/3.0 * ∇V[i,j,k]
    end

    @inbounds if i<size(εxx,1)+1 && j<size(εxx,2)+1 && k<size(εxx,3)+1
        if (i<=size(εxy,1)) && (j<=size(εxy,2)) && (k<=size(εxy,3))
            dVxΔy      = (Vx[i,j+1,k+1] - Vx[i,j,k+1]) *_Δy 
            dVyΔx      = (Vy[i+1,j,k+1] - Vy[i,j,k+1]) *_Δx 
            εxy[i,j,k] = 0.5*(dVxΔy + dVyΔx)
        end
        if (i<=size(εxz,1)) && (j<=size(εxz,2)) && (k<=size(εxz,3))
            dVxΔz      = (Vx[i  ,j+1,k+1] - Vx[i,j+1,k]) *_Δz                     
            dVzΔx      = (Vz[i+1,j+1,k  ] - Vz[i,j+1,k]) *_Δx 
            εxz[i,j,k] = 0.5*(dVxΔz + dVzΔx)
        end
        if (i<=size(εyz,1)) && (j<=size(εyz,2)) && (k<=size(εyz,3))
            dVyΔz      = (Vy[i+1,j,k+1] - Vy[i+1,j,k]) *_Δz 
            dVzΔy      = (Vz[i+1,j+1,k] - Vz[i+1,j,k]) *_Δy 
            εyz[i,j,k] = 0.5*(dVyΔz + dVzΔy)
        end
    end
    return
end

# ###############################

@tiny function Kernel_ComputeStressCenters!( τxx, τyy, τzz, τxyc, τxzc, τyzc, τII, λ̇, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, εII, ηc, Gc, Gv, P, Δt, pl, plasticity )
    i, j, k = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j, k)
    @inbounds if isin(εxx)
        ηe                = Gc[i,j,k]*Δt
        ηve               = ηc[i,j,k]
        εxxc              = εxx[i,j,k] + τxx0[i+1,j+1,k+1]/2.0/ηe
        εyyc              = εyy[i,j,k] + τyy0[i+1,j+1,k+1]/2.0/ηe
        εzzc              = εzz[i,j,k] + τzz0[i+1,j+1,k+1]/2.0/ηe
        εxyc              = 0.25*( (εxy[i,j,k] + τxy0[i,j,k]/2.0/Δt/Gv[i,j,k]) + (εxy[i+1,j,k] + τxy0[i+1,j,k]/2.0/Δt/Gv[i+1,j,k]) + (εxy[i,j+1,k] + τxy0[i,j+1,k]/2.0/Δt/Gv[i,j+1,k]) + (εxy[i+1,j+1,k] + τxy0[i+1,j+1,k]/2.0/Δt/Gv[i+1,j+1,k]))
        εxzc              = 0.25*( (εxz[i,j,k] + τxz0[i,j,k]/2.0/Δt/Gv[i,j,k]) + (εxz[i+1,j,k] + τxz0[i+1,j,k]/2.0/Δt/Gv[i+1,j,k]) + (εxz[i,j,k+1] + τxz0[i,j,k+1]/2.0/Δt/Gv[i,j,k+1]) + (εxz[i+1,j,k+1] + τxz0[i+1,j,k+1]/2.0/Δt/Gv[i+1,j,k+1]))
        εyzc              = 0.25*( (εyz[i,j,k] + τyz0[i,j,k]/2.0/Δt/Gv[i,j,k]) + (εyz[i,j+1,k] + τyz0[i,j+1,k]/2.0/Δt/Gv[i,j+1,k]) + (εyz[i,j,k+1] + τyz0[i,j,k+1]/2.0/Δt/Gv[i,j,k+1]) + (εyz[i,j+1,k+1] + τyz0[i,j+1,k+1]/2.0/Δt/Gv[i,j+1,k+1]))
        εII[i,j,k]        = sqrt( 0.5*(εxxc^2 + εyyc^2 + εzzc^2) + εxyc^2 + εxzc^2 + εyzc^2)
        τxx[i+1,j+1,k+1]  = 2.0*ηve*εxxc
        τyy[i+1,j+1,k+1]  = 2.0*ηve*εyyc
        τzz[i+1,j+1,k+1]  = 2.0*ηve*εzzc
        τxyc[i+1,j+1,k+1] = 2.0*ηve*εxyc
        τxzc[i+1,j+1,k+1] = 2.0*ηve*εxzc
        τyzc[i+1,j+1,k+1] = 2.0*ηve*εyzc

        if plasticity
            τII[i,j,k] = sqrt( 0.5*(τxx[i+1,j+1,k+1]^2 + τyy[i+1,j+1,k+1]^2 + τzz[i+1,j+1,k+1]^2) + τxyc[i+1,j+1,k+1]^2 + τxzc[i+1,j+1,k+1]^2 + τyzc[i+1,j+1,k+1]^2)
            λ̇[i,j,k]   = 0.
            F    = τII[i,j,k] - pl.C * cosd(pl.ϕ) - P[i+1,j+1,k+1] * sind(pl.ϕ) - pl.ηvp*λ̇[i,j,k]
            τIIc = τII[i,j,k]
            ηvep = ηve
            if F>0
                λ̇[i,j,k]   = F / (ηvep + pl.ηvp)
                τIIc       = pl.C * cosd(pl.ϕ) + P[i+1,j+1,k+1] * sind(pl.ϕ) + pl.ηvp*λ̇[i,j,k]
                ηvep       = τIIc / 2.0 ./ εII[i,j,k]
            end
            ηc[i,j,k]         = ηvep
            τxx[i+1,j+1,k+1]  = 2.0*ηvep*εxxc
            τyy[i+1,j+1,k+1]  = 2.0*ηvep*εyyc
            τzz[i+1,j+1,k+1]  = 2.0*ηvep*εzzc
            τxyc[i+1,j+1,k+1] = 2.0*ηvep*εxyc
            τxzc[i+1,j+1,k+1] = 2.0*ηvep*εxzc
            τyzc[i+1,j+1,k+1] = 2.0*ηvep*εyzc
        end

    end
    return
end

# ###############################

@tiny function Kernel_ComputePlasticity!( F, λ̇, τII, τxx, τyy, τzz, τxyc, τxzc, τyzc, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, εII, ηc, Gc, Gv, P, Δt, pl )
    i, j, k = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j, k)
    @inbounds if isin(F)

        ηe         = Gc[i,j,k]*Δt
        ηvep       = ηc[i,j,k]
        εxxc       = εxx[i,j,k] + τxx0[i+1,j+1,k+1]/2.0/ηe
        εyyc       = εyy[i,j,k] + τyy0[i+1,j+1,k+1]/2.0/ηe
        εzzc       = εzz[i,j,k] + τzz0[i+1,j+1,k+1]/2.0/ηe
        εxyc       = 0.25*( (εxy[i,j,k] + τxy0[i,j,k]/2.0/Δt/Gv[i,j,k]) + (εxy[i+1,j,k] + τxy0[i+1,j,k]/2.0/Δt/Gv[i+1,j,k]) + (εxy[i,j+1,k] + τxy0[i,j+1,k]/2.0/Δt/Gv[i,j+1,k]) + (εxy[i+1,j+1,k] + τxy0[i+1,j+1,k]/2.0/Δt/Gv[i+1,j+1,k]))
        εxzc       = 0.25*( (εxz[i,j,k] + τxz0[i,j,k]/2.0/Δt/Gv[i,j,k]) + (εxz[i+1,j,k] + τxz0[i+1,j,k]/2.0/Δt/Gv[i+1,j,k]) + (εxz[i,j,k+1] + τxz0[i,j,k+1]/2.0/Δt/Gv[i,j,k+1]) + (εxz[i+1,j,k+1] + τxz0[i+1,j,k+1]/2.0/Δt/Gv[i+1,j,k+1]))
        εyzc       = 0.25*( (εyz[i,j,k] + τyz0[i,j,k]/2.0/Δt/Gv[i,j,k]) + (εyz[i,j+1,k] + τyz0[i,j+1,k]/2.0/Δt/Gv[i,j+1,k]) + (εyz[i,j,k+1] + τyz0[i,j,k+1]/2.0/Δt/Gv[i,j,k+1]) + (εyz[i,j+1,k+1] + τyz0[i,j+1,k+1]/2.0/Δt/Gv[i,j+1,k+1]))
        εII[i,j,k] = sqrt( 0.5*(εxxc^2 + εyyc^2 + εzzc^2) + εxyc^2 + εxzc^2 + εyzc^2)
        τII[i,j,k] = sqrt( 0.5*(τxx[i+1,j+1,k+1]^2 + τyy[i+1,j+1,k+1]^2 + τzz[i+1,j+1,k+1]^2) + τxyc[i+1,j+1,k+1]^2 + τxzc[i+1,j+1,k+1]^2 + τyzc[i+1,j+1,k+1]^2)
        λ̇[i,j,k]   = 0.
        F[i,j,k]   = τII[i,j,k] - pl.C * cosd(pl.ϕ) - P[i+1,j+1,k+1] * sind(pl.ϕ) - pl.ηvp*λ̇[i,j,k]

        τIIc = τII[i,j,k]
        if F[i,j,k]>0
            λ̇[i,j,k]   = F[i,j,k] / (ηvep + pl.ηvp)
            τIIc       = pl.C * cosd(pl.ϕ) + P[i+1,j+1,k+1] * sind(pl.ϕ) + pl.ηvp*λ̇[i,j,k]
            ηvep       = τIIc / 2.0 ./ εII[i,j,k]
        end

        ηc[i,j,k]         = ηvep
        # τxx[i+1,j+1,k+1]  = 2.0*ηvep*εxxc
        # τyy[i+1,j+1,k+1]  = 2.0*ηvep*εyyc
        # τzz[i+1,j+1,k+1]  = 2.0*ηvep*εzzc
        # τxyc[i+1,j+1,k+1] = 2.0*ηvep*εxyc
        # τxzc[i+1,j+1,k+1] = 2.0*ηvep*εxzc
        # τyzc[i+1,j+1,k+1] = 2.0*ηvep*εyzc
        τxx[i+1,j+1,k+1]  = τy
        τyy[i+1,j+1,k+1]  = 2.0*ηvep*εyyc
        τzz[i+1,j+1,k+1]  = 2.0*ηvep*εzzc
        τxyc[i+1,j+1,k+1] = 2.0*ηvep*εxyc
        τxzc[i+1,j+1,k+1] = 2.0*ηvep*εxzc
        τyzc[i+1,j+1,k+1] = 2.0*ηvep*εyzc
    end
    return
end

@tiny function Kernel_CheckYield!(F, λ̇, τII, P, τxx, τyy, τzz, τxyc, τxzc, τyzc, pl)
    i, j, k = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j, k)
    @inbounds if isin(F)
        τII[i,j,k] = sqrt( 0.5*(τxx[i+1,j+1,k+1]^2 + τyy[i+1,j+1,k+1]^2 + τzz[i+1,j+1,k+1]^2) + τxyc[i+1,j+1,k+1]^2 + τxzc[i+1,j+1,k+1]^2 + τyzc[i+1,j+1,k+1]^2)
        F[i,j,k]   = τII[i,j,k] - pl.C * cosd(pl.ϕ) - P[i+1,j+1,k+1] * sind(pl.ϕ) - pl.ηvp*λ̇[i,j,k]
    end
    return
end

# ###############################

@tiny function Kernel_InterpShearStress!( τxyc, τxzc, τyzc, τxy, τxz, τyz)
    i, j, k = @indices
    # Free slip: Keep zero values where needed
    @inbounds if i>1 && i<size(τxy,1) && j>1 && j<size(τxy,2) && k>=1 && k<=size(τxy,3)
        τxy[i,j,k] = 0.25*(τxyc[i,j,k+1] + τxyc[i+1,j,k+1] + τxyc[i,j+1,k+1] + τxyc[i+1,j+1,k+1])
    end
    @inbounds if i>1 && i<size(τxz,1) && j>=1 && j<=size(τxz,2) && k>1 && k<size(τxz,3) 
        τxz[i,j,k] = 0.25*(τxzc[i,j+1,k] + τxzc[i+1,j+1,k] + τxzc[i,j+1,k+1] + τxzc[i+1,j+1,k+1])
    end
    @inbounds if i>=1 && i<=size(τyz,1) && j>1 && j<size(τyz,2) && k>1 && k<size(τyz,3) 
        τyz[i,j,k] = 0.25*(τyzc[i+1,j,k] + τyzc[i+1,j+1,k] + τyzc[i+1,j,k+1] + τyzc[i+1,j+1,k+1])
    end
    return
end

# ###############################

@tiny function Kernel_ComputeResiduals!( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, P0, K, ∇V, Δx, Δy, Δz, Δt )
    _Δx, _Δy, _Δz, _KΔt = 1.0/Δx, 1.0/Δy, 1.0/Δz, 1.0/(K*Δt)
    i, j, k = @indices
    @inbounds if i<size(P,1)-1 && j<size(P,2)-1 && k<size(P,3)-1
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
            Fp[i,j,k] = -∇V[i,j,k] - (P[i+1,j+1,k+1] - P0[i+1,j+1,k+1]) *_KΔt
        end
    end
    return
end

# ###############################

@tiny function Kernel_UpdateRates!( dVxdτ, dVydτ, dVzdτ, dPdτ, Fx, Fy, Fz, Fp, ch_ρ, h_ρ, ncx, ncy, ncz )
    i, j, k = @indices
    α = (2-ch_ρ)/(2+ch_ρ)
    β =    2*h_ρ/(2+ch_ρ)
    @inbounds if i<ncx+1 && j<ncy+1 && k<ncz+1
        if (i<=size(Fx,1)) && (j<=size(Fx,2)) && (k<=size(Fx,3))
            dVxdτ[i,j,k] = α*dVxdτ[i,j,k] + β*Fx[i,j,k]
        end

        if (i<=size(Fy,1)) && (j<=size(Fy,2)) && (k<=size(Fy,3))
            dVydτ[i,j,k] = α*dVydτ[i,j,k] + β*Fy[i,j,k]
        end

        if (i<=size(Fz,1)) && (j<=size(Fz,2)) && (k<=size(Fz,3))
            dVzdτ[i,j,k] = α*dVzdτ[i,j,k] + β*Fz[i,j,k]
        end

        if (i<=size(Fp,1)) && (j<=size(Fp,2)) && (k<=size(Fp,3))
            dPdτ[i,j,k] = α*dPdτ[i,j,k]   + β*Fp[i,j,k]
        end
    end
    return
end

# ###############################

@tiny function Kernel_UpdateVP!( dVxdτ, dVydτ, dVzdτ, dPdτ, Vx, Vy, Vz, P, hx, hy, hz, hp, ncx, ncy, ncz )
    i, j, k = @indices
    @inbounds if i<ncx+1 && j<ncy+1 && k<ncz+1
        if (i<=size(dVxdτ,1)) && (j<=size(dVxdτ,2)) && (k<=size(dVxdτ,3))
            Vx[i,j+1,k+1] += hx[i,j+1,k+1]*dVxdτ[i,j,k]
        end

        if (i<=size(dVydτ,1)) && (j<=size(dVydτ,2)) && (k<=size(dVydτ,3))
            Vy[i+1,j,k+1] += hy[i+1,j,k+1]*dVydτ[i,j,k]
        end

        if (i<=size(dVzdτ,1)) && (j<=size(dVzdτ,2)) && (k<=size(dVzdτ,3))
            Vz[i+1,j+1,k] += hz[i+1,j+1,k]*dVzdτ[i,j,k]
        end

        if (i<=size(dPdτ,1)) && (j<=size(dPdτ,2)) && (k<=size(dPdτ,3))
            P[i+1,j+1,k+1] += hp[i+1,j+1,k+1]*dPdτ[i,j,k]
        end
    end
    return
end

#############################################################################################
#############################################################################################
#############################################################################################

function Stokes3D(n, ::Type{DAT}; device) where DAT
    Lx,  Ly,  Lz  =  1.0,  1.0,  0.8150  
    ncx, ncy, ncz = n*32, 3, n*32
    BCtype = :PureShear_xz
    ε      = 5e-6/1e4
    r      = 5e-2
    K      = 2.0
    pl     =  (ϕ      = 30.0, ψ      = 10.0, C      = 1.75e-4, ηvp   = 2.5e2)
    
    Δt     = 1e4
    t      = 0.0
    nt     = 30

    write_out    = true
    write_nout   = 1
    restart_from = 0
    visu         = true

    # --------- DYREL --------- #
    niter  = 1e5
    nout   = 1
    tol    = 1e-8

    #-----------
    # Load benchmark
    file = matopen(string("data/DataM2Di_EVP_model1.mat"))
    τ_bench = read(file, "Tiivec") # note that this does NOT introduce a variable ``varname`` into scope
    close(file)
    #-----------
    P     = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(P,  DAT(0.))
    P0    = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(P0, DAT(0.))
    Vx    = device_array(DAT, device, ncx+1, ncy+2, ncz+2); fill!(Vx, DAT(0.))
    Vy    = device_array(DAT, device, ncx+2, ncy+1, ncz+2); fill!(Vy, DAT(0.))
    Vz    = device_array(DAT, device, ncx+2, ncy+2, ncz+1); fill!(Vz, DAT(0.))
    ηc    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(ηc, DAT(0.))
    ηv    = device_array(DAT, device, ncx+1, ncy+1, ncz+1); fill!(ηv, DAT(0.))
    Gc    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(ηc, DAT(0.))
    Gv    = device_array(DAT, device, ncx+1, ncy+1, ncz+1); fill!(ηv, DAT(0.))
    τxx   = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τxx, DAT(0.))
    τyy   = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τyy, DAT(0.))
    τzz   = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τzz, DAT(0.))
    τxy   = device_array(DAT, device, ncx+1, ncy+1, ncz+0); fill!(τxy, DAT(0.))
    τxz   = device_array(DAT, device, ncx+1, ncy+0, ncz+1); fill!(τxz, DAT(0.))
    τyz   = device_array(DAT, device, ncx+0, ncy+1, ncz+1); fill!(τyz, DAT(0.))

    τxyc  = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τxyc, DAT(0.))
    τxzc  = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τxzc, DAT(0.))
    τyzc  = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τyzc, DAT(0.))
    
    F    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(F, DAT(0.))
    λ̇    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(λ̇, DAT(0.))
   
    τII   = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(τII, DAT(0.))
    τxx0  = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τxx0, DAT(0.))
    τyy0  = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τyy0, DAT(0.))
    τzz0  = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τzz0, DAT(0.))
    τxy0  = device_array(DAT, device, ncx+1, ncy+1, ncz+0); fill!(τxy0, DAT(0.))
    τxz0  = device_array(DAT, device, ncx+1, ncy+0, ncz+1); fill!(τxz0, DAT(0.))
    τyz0  = device_array(DAT, device, ncx+0, ncy+1, ncz+1); fill!(τyz0, DAT(0.))
    
    ∇V    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(∇V,  DAT(0.))
    εII   = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(εII, DAT(0.))
    εxx   = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(εxx, DAT(0.))
    εyy   = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(εyy, DAT(0.))
    εzz   = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(εzz, DAT(0.))
    εxy   = device_array(DAT, device, ncx+1, ncy+1, ncz+0); fill!(εxy, DAT(0.))
    εxz   = device_array(DAT, device, ncx+1, ncy+0, ncz+1); fill!(εxz, DAT(0.))
    εyz   = device_array(DAT, device, ncx+0, ncy+1, ncz+1); fill!(εyz, DAT(0.))
    Fx    = device_array(DAT, device, ncx+1, ncy+0, ncz+0); fill!(Fx,  DAT(0.))
    Fy    = device_array(DAT, device, ncx+0, ncy+1, ncz+0); fill!(Fy,  DAT(0.))
    Fz    = device_array(DAT, device, ncx+0, ncy+0, ncz+1); fill!(Fz,  DAT(0.))
    Fp    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(Fp,  DAT(0.))
    dVxdτ = device_array(DAT, device, ncx+1, ncy+0, ncz+0); fill!(dVxdτ, DAT(0.))
    dVydτ = device_array(DAT, device, ncx+0, ncy+1, ncz+0); fill!(dVydτ, DAT(0.))
    dVzdτ = device_array(DAT, device, ncx+0, ncy+0, ncz+1); fill!(dVzdτ, DAT(0.))
    dPdτ  = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(dPdτ,  DAT(0.))

    hx    = device_array(DAT, device, ncx+1, ncy+2, ncz+2); fill!(hx,  DAT(0.))
    hy    = device_array(DAT, device, ncx+2, ncy+1, ncz+2); fill!(hy,  DAT(0.))
    hz    = device_array(DAT, device, ncx+2, ncy+2, ncz+1); fill!(hz,  DAT(0.))
    hp    = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(hp,  DAT(0.))
    
    λx    = device_array(DAT, device, ncx+1, ncy+0, ncz+0); fill!(λx,  DAT(0.))
    λy    = device_array(DAT, device, ncx+0, ncy+1, ncz+0); fill!(λy,  DAT(0.))
    λz    = device_array(DAT, device, ncx+0, ncy+0, ncz+1); fill!(λz,  DAT(0.))
    λp    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(λp,  DAT(0.))

    δx    = device_array(DAT, device, ncx+1, ncy+2, ncz+2); fill!(δx,  DAT(0.))
    δy    = device_array(DAT, device, ncx+2, ncy+1, ncz+2); fill!(δy,  DAT(0.))
    δz    = device_array(DAT, device, ncx+2, ncy+2, ncz+1); fill!(δz,  DAT(0.))
    δp    = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(δp,  DAT(0.))

    τii_vec = zeros(nt)
    t_vec   = zeros(nt)


    #-----------
    xv = LinRange(-Lx/2, Lx/2, ncx+1)
    yv = LinRange(-Ly/2, Ly/2, ncy+1)
    zv = LinRange(-Lz/2, Lz/2, ncz+1)
    Δx, Δy, Δz = Lx/ncx, Ly/ncy, Lz/ncz
    xce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2)
    yce = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    zce = LinRange(-Lz/2-Δz/2, Lz/2+Δz/2, ncz+2)

    # Kernels
    InitialCondition!     = Kernel_InitialCondition!(device)
    SetBoundaries!        = Kernel_SetBoundaries!(device)
    InterpV2C!            = Kernel_InterpV2C!(device)
    ComputeStrainRates!   = Kernel_ComputeStrainRates!(device)
    ComputeStressCenters! = Kernel_ComputeStressCenters!(device)
    ComputeResiduals!     = Kernel_ComputeResiduals!(device)
    UpdateRates!          = Kernel_UpdateRates!(device)
    UpdateVP!             = Kernel_UpdateVP!(device)
    Gerschgorin1!         = Kernel_Gerschgorin1!(device)
    Gerschgorin2!         = Kernel_Gerschgorin2!(device)
    Gerschgorin2b!        = Kernel_Gerschgorin2b!(device)
    Gerschgorin3!         = Kernel_Gerschgorin3!(device)
    MaxLoc!               = Kernel_MaxLoc!(device)
    InterpShearStress!    = Kernel_InterpShearStress!(device)
    CheckYield!           = Kernel_CheckYield!(device)
    TinyKernels.device_synchronize(device)
    
    wait( InitialCondition!( Vx, Vy, Vz, ηv, ηc, Gv, Gc, ε, xv, yv, zv, xce, yce, zce, r; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    
    ηv .= 1.0 ./(1.0./ηv .+ 1.0./(Gv.*Δt))
    
    wait( InterpV2C!( ηc, ηv; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    wait( InterpV2C!( Gc, Gv; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    ηv2   = device_array(DAT, device, ncx+1, ncy+1, ncz+1); fill!(ηv2, DAT(0.))
    ηv2  .= ηv
    wait( MaxLoc!(ηv2, ηv; ndrange=(ncx+2, ncy+2, ncz+2)) )

    ηc2   = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(ηc2, DAT(0.))
    ηc2  .= ηc
    wait( MaxLoc!(ηc2, ηc; ndrange=(ncx+2, ncy+2, ncz+2)) )

    # --------- DYREL --------- #

    # Steps
    hx[2:end-1,2:end-1,2:end-1] .= 1.0 ./av2x_arit(ηc2)*4
    hy[2:end-1,2:end-1,2:end-1] .= 1.0 ./av2y_arit(ηc2)*4
    hz[2:end-1,2:end-1,2:end-1] .= 1.0 ./av2z_arit(ηc2)*4
    hx[[1, end],:,:] .=  hx[[2, end-1],:,:];  hx[:,[1, end],:] .=  hx[:,[2, end-1],:];  hx[:,:,[1, end]] .=  hx[:,:,[2, end-1]]
    hy[[1, end],:,:] .=  hy[[2, end-1],:,:];  hy[:,[1, end],:] .=  hy[:,[2, end-1],:];  hy[:,:,[1, end]] .=  hy[:,:,[2, end-1]]
    hz[[1, end],:,:] .=  hz[[2, end-1],:,:];  hz[:,[1, end],:] .=  hz[:,[2, end-1],:];  hz[:,:,[1, end]] .=  hz[:,:,[2, end-1]]
    hp    .= 1.0  

    # Gerschgorin
    wait( Gerschgorin1!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Δx, Δy, Δz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    wait( Gerschgorin2!( τxx, τyy, τzz, τxyc, τxzc, τyzc, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, ηc, Gc, Gv, P, Δt, pl; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    wait( Gerschgorin2b!( τxyc, τxzc, τyzc, τxy, τxz, τyz; ndrange=(ncx+2, ncy+2, ncz+2)) )
    wait( Gerschgorin3!( λx, λy, λz, λp, τxx, τyy, τzz, τxy, τxz, τyz, P, P0, K, ∇V, Δx, Δy, Δz, Δt; ndrange=(ncx+2, ncy+2, ncz+2) ) ) 
    @show λmax =  maximum([maximum(λx.*hx[:,2:end-1,2:end-1]) maximum(λy.*hy[2:end-1,:,2:end-1]) maximum(λz.*hz[2:end-1,2:end-1,:]) maximum(λp.*hp[2:end-1,2:end-1,2:end-1])])

    @show λmin = 1.0
    h_ρ   = 4.0/(λmin + λmax)
    ch_ρ  = 4.0*sqrt(λmin*λmax)/(λmin + λmax)

    τxx .= 0.0
    τyy .= 0.0
    τzz .= 0.0
    τxy .= 0.0
    τxz .= 0.0
    τyz .= 0.0

    #---------------------------------------------#
    # Breakpoint business
    it0 = 1
    if restart_from > 0
        fname = @sprintf("./Breakpoint%05d.h5", restart_from)
        @printf("Reading file %s\n", fname)
        h5open(fname, "r") do file
            Vx    .= to_device(read(file, "Vx")   )
            Vy    .= to_device(read(file, "Vy")   )
            Vz    .= to_device(read(file, "Vz")   )
            P     .= to_device(read(file, "P")    )
            τxx   .= to_device(read(file, "Txx")  )
            τyy   .= to_device(read(file, "Tyy")  )
            τzz   .= to_device(read(file, "Tzz")  )
            τxy   .= to_device(read(file, "Txy")  )
            τxz   .= to_device(read(file, "Txz")  )
            τyz   .= to_device(read(file, "Tyz")  )
            dVxdτ .= to_device(read(file, "dVxdt"))
            dVydτ .= to_device(read(file, "dVydt"))
            dVzdτ .= to_device(read(file, "dVzdt"))
            dPdτ  .= to_device(read(file, "dPdt") )
        end
        it0 = restart_from+1
    end


    # --------- DYREL --------- #

    for it=it0:nt

        @printf("############# Step %04d #############\n", it)

        τxx0 .= τxx  
        τyy0 .= τyy  
        τzz0 .= τzz  
        τxy0 .= τxy  
        τxz0 .= τxz  
        τyz0 .= τyz 
        P0   .= P

        for iter=1:niter

            wait( SetBoundaries!( Vx, Vy, Vz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
            wait( ComputeStrainRates!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Vx, Vy, Vz, Δx, Δy, Δz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
            wait( ComputeStressCenters!( τxx, τyy, τzz, τxyc, τxzc, τyzc, τII, λ̇, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, εII, ηc, Gc, Gv, P, Δt, pl, true; ndrange=(ncx+2, ncy+2, ncz+2) ) )
            # wait( ComputePlasticity!( F, λ̇, τII, τxx, τyy, τzz, τxyc, τxzc, τyzc, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, εII, ηc, Gc, Gv, P, Δt, pl; ndrange=(ncx+2, ncy+2, ncz+2) ) )
            wait( InterpShearStress!( τxyc, τxzc, τyzc, τxy, τxz, τyz; ndrange=(ncx+2, ncy+2, ncz+2)) )
            wait( ComputeResiduals!( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, P0, K, ∇V, Δx, Δy, Δz, Δt; ndrange=(ncx+2, ncy+2, ncz+2) ) )  
            wait( UpdateRates!( dVxdτ, dVydτ, dVzdτ, dPdτ, Fx, Fy, Fz, Fp, ch_ρ, h_ρ, ncx, ncy, ncz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
            wait( UpdateVP!( dVxdτ, dVydτ, dVzdτ, dPdτ, Vx, Vy, Vz, P, hx, hy, hz, hp, ncx, ncy, ncz; ndrange=(ncx+2, ncy+2, ncz+2) ) )

            if mod(iter,nout) == 0 || iter==1
                CheckYield!(F, λ̇, τII, P, τxx, τyy, τzz, τxyc, τxzc, τyzc, pl; ndrange=(ncx+2, ncy+2, ncz+2) )
                nFx = norm(Fx)/sqrt(length(Fx))
                nFy = norm(Fy)/sqrt(length(Fy))
                nFz = norm(Fz)/sqrt(length(Fz))
                nFp = norm(Fp)/sqrt(length(Fp))
                @printf("Iter. %05d\n", iter) 
                @printf("Fx = %1.4e\n", nFx) 
                @printf("Fy = %1.4e\n", nFy) 
                @printf("Fz = %1.4e\n", nFz) 
                @printf("Fp = %1.4e\n", nFp) 
                @printf("max(F) = %1.4e --- mean(τII) = %1.4e --- t = %1.4e\n", maximum(F), mean(τII), it*Δt) 
                max(nFx, nFy, nFz, nFp)<tol && break # short circuiting operations
                isnan(nFx) && error("NaN emergency!") 
                nFx>1e8    && error("Blow up!") 
                # Adapt pseudo time steps
                ηc2  .= ηc
                wait( MaxLoc!(ηc2, ηc; ndrange=(ncx+2, ncy+2, ncz+2)) )
                hx[2:end-1,2:end-1,2:end-1] .= 1.0 ./av2x_arit(ηc2)*4
                hy[2:end-1,2:end-1,2:end-1] .= 1.0 ./av2y_arit(ηc2)*4
                hz[2:end-1,2:end-1,2:end-1] .= 1.0 ./av2z_arit(ηc2)*4
                hx[[1, end],:,:] .=  hx[[2, end-1],:,:];  hx[:,[1, end],:] .=  hx[:,[2, end-1],:];  hx[:,:,[1, end]] .=  hx[:,:,[2, end-1]]
                hy[[1, end],:,:] .=  hy[[2, end-1],:,:];  hy[:,[1, end],:] .=  hy[:,[2, end-1],:];  hy[:,:,[1, end]] .=  hy[:,:,[2, end-1]]
                hz[[1, end],:,:] .=  hz[[2, end-1],:,:];  hz[:,[1, end],:] .=  hz[:,[2, end-1],:];  hz[:,:,[1, end]] .=  hz[:,:,[2, end-1]]
                hp               .= 1.0  
                # Gerschgorin
                wait( Gerschgorin1!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Δx, Δy, Δz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
                wait( Gerschgorin2!( τxx, τyy, τzz, τxyc, τxzc, τyzc, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, ηc, Gc, Gv, P, Δt, pl; ndrange=(ncx+2, ncy+2, ncz+2) ) )
                wait( Gerschgorin2b!( τxyc, τxzc, τyzc, τxy, τxz, τyz; ndrange=(ncx+2, ncy+2, ncz+2)) )
                wait( Gerschgorin3!( λx, λy, λz, λp, τxx, τyy, τzz, τxy, τxz, τyz, P, P0, K, ∇V, Δx, Δy, Δz, Δt; ndrange=(ncx+2, ncy+2, ncz+2) ) ) 
                λmax =  maximum([maximum(λx.*hx[:,2:end-1,2:end-1]) maximum(λy.*hy[2:end-1,:,2:end-1]) maximum(λz.*hz[2:end-1,2:end-1,:]) maximum(λp.*hp[2:end-1,2:end-1,2:end-1])])
                # Joldes et al. (2011) --- v4 
                δx[:,2:end-1,2:end-1]       .= hx[:,2:end-1,2:end-1].*dVxdτ
                δy[2:end-1,:,2:end-1]       .= hy[2:end-1,:,2:end-1].*dVydτ
                δz[2:end-1,2:end-1,:]       .= hz[2:end-1,2:end-1,:].*dVzdτ
                δp[2:end-1,2:end-1,2:end-1] .= hp[2:end-1,2:end-1,2:end-1].*dPdτ
                # Input: -hi.*δi,  Output: Field
                # Make sure no force fector is accounted: 0.0.*τij
                # No plasticity (false), just evaluate with already computed effective viscosity
                wait( ComputeStrainRates!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, -hx.*δx, -hy.*δy, -hz.*δz, Δx, Δy, Δz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
                wait( ComputeStressCenters!( τxx, τyy, τzz, τxyc, τxzc, τyzc, τII, λ̇,  0.0.*τxx0,  0.0.*τyy0,  0.0.*τzz0,  0.0.*τxy0,  0.0.*τxz0,  0.0.*τyz0, εxx, εyy, εzz, εxy, εxz, εyz, εII, ηc, Gc, Gv, P, Δt, pl, false; ndrange=(ncx+2, ncy+2, ncz+2) ) )
                wait( InterpShearStress!( τxyc, τxzc, τyzc, τxy, τxz, τyz; ndrange=(ncx+2, ncy+2, ncz+2)) )
                wait( ComputeResiduals!( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, -hp.*δp, 0.0.*P0, K, ∇V, Δx, Δy, Δz, Δt; ndrange=(ncx+2, ncy+2, ncz+2) ) )  
                
                # @show norm(∇V)
                # @show norm(εxx)
                # @show norm(εyy)
                # @show norm(εzz)
                # @show norm(εxy)
                # @show norm(τxx)
                # @show norm(τyy)
                # @show norm(τzz)
                # @show norm(τxyc)
                # @show norm(τxzc)
                # @show norm(τxzc)
                # @show norm(τxy)
                # @show norm(τxz)
                # @show norm(τxz)
                # @show norm(Fx)
                # @show norm(Fy)
                # @show norm(Fz)
                # @show norm(Fp)
                
                λmin0 = λmin
                λmin  = (sum(δx[:,2:end-1,2:end-1].*Fx) + sum(δy[2:end-1,:,2:end-1].*Fy) + sum(δz[2:end-1,2:end-1,:].*Fz) + sum(δp[2:end-1,2:end-1,2:end-1].*Fp)) / (sum(δx[:,2:end-1,2:end-1].*δx[:,2:end-1,2:end-1]) + sum(δy[2:end-1,:,2:end-1].*δy[2:end-1,:,2:end-1]) + sum(δz[2:end-1,2:end-1,:].*δz[2:end-1,2:end-1,:]) + sum(δp[2:end-1,2:end-1,2:end-1].*δp[2:end-1,2:end-1,2:end-1]))
                if λmin<0. λmin = λmin0 end
                # Adapt optimal parameters
                h_ρ   = 4.0/(λmin + λmax)
                ch_ρ  = 4.0*sqrt(λmin*λmax)/(λmin + λmax)
                @show  (λmax, λmin)
            end
        end

        t          += Δt
        τii_vec[it] = mean(τII)
        t_vec[it]   = t

        if visu
            fig = Figure(resolution = ( Lx/Lz*800,800), fontsize=25, aspect = 2.0)
            ax = Axis(fig[1, 1], title = "Field", xlabel = "x [m]", ylabel = "y [m]")
            lines!(t_vec[1:it], τ_bench[1:it])
            scatter!(ax, t_vec[1:it], τii_vec[1:it])
            ax = Axis(fig[2, 1])
            # heatmap!(ax, to_host(ηv[:, size(τxy,2)÷2, :]))
            hm = heatmap!(ax, xce[2:end-1], zce[2:end-1], to_host(P[2:end-1, 2, 2:end-1]), colormap=cgrad(:roma, rev=true))
            Colorbar(fig[2, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
            # hm = heatmap!(ax, xce[2:end-1], zce[2:end-1], to_host(τII), colormap=cgrad(:roma, rev=true))
            # Colorbar(fig[2, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
            DataInspector(fig)
            display(fig)
        end

        #---------------------------------------------#
        # Breakpoint business
        if write_out==true && (it==1 || mod(it, write_nout)==0)
            fname = @sprintf("./Breakpoint%05d.h5", it)
            @printf("Writing file %s\n", fname)
            h5open(fname, "w") do file
                write(file, "Vx"   , to_host(Vx   ))
                write(file, "Vy"   , to_host(Vy   ))
                write(file, "Vz"   , to_host(Vz   ))
                write(file, "P"    , to_host(P    ))
                write(file, "Txx"  , to_host(τxx  ))
                write(file, "Tyy"  , to_host(τyy  ))
                write(file, "Tzz"  , to_host(τzz  ))
                write(file, "Txy"  , to_host(τxy  ))
                write(file, "Txz"  , to_host(τxz  ))
                write(file, "Tyz"  , to_host(τyz  ))
                write(file, "dVxdt", to_host(dVxdτ))
                write(file, "dVydt", to_host(dVydτ))
                write(file, "dVzdt", to_host(dVzdτ))
                write(file, "dPdt" , to_host(dPdτ ))
            end
        end

        @printf("τxx : min = %2.4e --- max = %2.4e\n", minimum(τxx[2:end-1,2:end-1,2:end-1]), maximum(τxx[2:end-1,2:end-1,2:end-1]))
        @printf("τyy : min = %2.4e --- max = %2.4e\n", minimum(τyy[2:end-1,2:end-1,2:end-1]), maximum(τyy[2:end-1,2:end-1,2:end-1]))
        @printf("τzz : min = %2.4e --- max = %2.4e\n", minimum(τzz[2:end-1,2:end-1,2:end-1]), maximum(τzz[2:end-1,2:end-1,2:end-1]))
        @printf("τxy : min = %2.4e --- max = %2.4e\n", minimum(τxyc[2:end-1,2:end-1,2:end-1]), maximum(τxyc[2:end-1,2:end-1,2:end-1]))
        @printf("τxy : min = %2.4e --- max = %2.4e\n", minimum(τxy), maximum(τxy))
        @printf("τxz : min = %2.4e --- max = %2.4e\n", minimum(τxzc[2:end-1,2:end-1,2:end-1]), maximum(τxzc[2:end-1,2:end-1,2:end-1]))
        @printf("τxz : min = %2.4e --- max = %2.4e\n", minimum(τxz), maximum(τxz))
        @printf("τyz : min = %2.4e --- max = %2.4e\n", minimum(τyzc[2:end-1,2:end-1,2:end-1]), maximum(τyzc[2:end-1,2:end-1,2:end-1]))
        @printf("τyz : min = %2.4e --- max = %2.4e\n", minimum(τyz), maximum(τyz))
        @printf("τII : min = %2.4e --- max = %2.4e\n", minimum(τII), maximum(τII))
        @printf("P0  : min = %2.4e --- max = %2.4e\n", minimum( P0[2:end-1,2:end-1,2:end-1]), maximum( P0[2:end-1,2:end-1,2:end-1]))
        @printf("P   : min = %2.4e --- max = %2.4e\n", minimum(  P[2:end-1,2:end-1,2:end-1]), maximum(  P[2:end-1,2:end-1,2:end-1]))
        @printf("ηc  : min = %2.4e --- max = %2.4e\n", minimum( ηc[2:end-1,2:end-1,2:end-1]), maximum( ηc[2:end-1,2:end-1,2:end-1]))
        @printf("ηv  : min = %2.4e --- max = %2.4e\n", minimum(ηv), maximum(ηv))

    end

    Vxc            = to_host(0.5.*(Vx[1:end-1,2:end-1,2:end-1] .+ Vx[2:end-0,2:end-1,2:end-1]))
    Vyc            = to_host(0.5.*(Vy[2:end-1,1:end-1,2:end-1] .+ Vy[2:end-1,2:end-0,2:end-1]))
    Vzc            = to_host(0.5.*(Vz[2:end-1,2:end-1,1:end-1] .+ Vz[2:end-1,2:end-1,2:end-0]))
    Vc             = (Vxc, Vyc, Vzc)

    fname = @sprintf("Output")
    vtk_grid(fname, Array(xce[2:end-1]), Array(yce[2:end-1]), Array(zce[2:end-1])) do vtk
        vtk["V"]   = Vc
        vtk["P"]   = to_host(P[2:end-1,2:end-1,2:end-1])
        vtk["η"]   = to_host(ηc)
        vtk["λ̇"]   = to_host(λ̇)
    end
    
    #-----------
    return nothing
end

#############################################################################################
#############################################################################################
#############################################################################################

@time Stokes3D( 2, eletype; device )
