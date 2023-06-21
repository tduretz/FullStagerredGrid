# 3D Stokes taken from PT3D - https://github.com/tduretz/PT3D/blob/main/Stokes3D_Threads_v0.jl
# Use TinyKerkels instead of plain Julia
# DYREL
using TinyKernels, Printf, WriteVTK
import LinearAlgebra: norm
import Statistics: mean

include("setup_example.jl")

# Select based upon your local device (:CPU, :CUDA, :AMDGPU, :Metal)
backend = :CUDA

include("helpers.jl") # will be defined in TinyKernels soon
include("Gershgorin3DSSG.jl")

@setup_example()

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
   
@tiny function Kernel_InitialCondition!( Vx, Vy, Vz, ηv, ηc, ε, xv, yv, zv, xce, yce, zce, rc )
    i, j, k = @indices
    @inbounds if i<length(xce) && k<length(zce) && j<length(yce)
        if (i<=size(Vx,1)) Vx[i,j,k] = -ε*xv[i] end
        if (k<=size(Vz,3)) Vz[i,j,k] =  ε*zv[k] end
        if (i<=size(ηv,1) && j<=size(ηv,2) && k<=size(ηv,3)) 
            ηv[i,j,k] = 1.0 
            x = xv[i] - 0.3
            y = yv[j]
            z = zv[k] 
            r = rc/2
            if ( (x*x + y*y + z*z) < r*r )  ηv[i,j,k] = 1000.0 end  

            x = xv[i] - 0.2
            y = yv[j] + 0.11
            z = zv[k] 
            r  = rc/2
            if ( (x*x + y*y + z*z) < r*r )  ηv[i,j,k] = 500.0 end 

            x = xv[i] + 0.2
            y = yv[j] + 0.11
            z = zv[k] - 0.4
            r  = rc/2
            if ( (x*x + y*y + z*z) < r*r )  ηv[i,j,k] = 50.0 end 

            x = xv[i] + 0.2
            y = yv[j] - 0.11
            z = zv[k] - 0.4
            r  = rc/1.5
            if ( (x*x + y*y + z*z) < r*r )  ηv[i,j,k] = 5e-2 end 

            x = xv[i] - 0.3
            y = yv[j] - 0.3
            z = zv[k] - 0.3
            r  = rc
            if ( (x*x + y*y + z*z) < r*r )  ηv[i,j,k] = 5e-2 end 

            x = xv[i] + 0.3
            y = yv[j] + 0.3
            z = zv[k] + 0.3
            r  = rc*1.1
            if ( (x*x + y*y + z*z) < r*r )  ηv[i,j,k] = 1e2 end 

            x = xv[i] + 0.3
            y = yv[j] - 0.25
            z = zv[k] + 0.3
            r  = rc*0.7
            if ( (x*x + y*y + z*z) < r*r )  ηv[i,j,k] = 1e-2 end 
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
        if j==size(Vz,2) Vx[i,j,k] = Vx[i,j-1,k] end
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
        if k==size(Vx,3) Vy[i,j,k] = Vy[i,j,k-1] end
    end
    # Vz
    @inbounds if isin(Vz)
        # X: West / East
        if i==1          Vz[i,j,k] = Vz[i+1,j,k] end
        if i==size(Vy,1) Vz[i,j,k] = Vz[i-1,j,k] end
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

@tiny function Kernel_InterpV2xyz!( ηxy, ηxz, ηyz, ηv )
    i, j, k = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j, k)
    @inbounds if isin(ηv)
        if (k<=size(ηxy,3)) ηxy[i,j,k]  = 1.0/2.0*(ηv[i,j,k] + ηv[i,j,k+1]) end
        if (j<=size(ηxz,2)) ηxz[i,j,k]  = 1.0/2.0*(ηv[i,j,k] + ηv[i,j+1,k]) end
        if (i<=size(ηyz,1)) ηyz[i,j,k]  = 1.0/2.0*(ηv[i,j,k] + ηv[i+1,j,k]) end
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

@tiny function Kernel_ComputeStress!( τxx, τyy, τzz, τxy, τxz, τyz, εxx, εyy, εzz, εxy, εxz, εyz, ηc, ηxy, ηxz, ηyz )
    i, j, k = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j, k)
    @inbounds if isin(εxx)
        τxx[i+1,j+1,k+1] = 2*ηc[i,j,k]*εxx[i,j,k]
        τyy[i+1,j+1,k+1] = 2*ηc[i,j,k]*εyy[i,j,k]
        τzz[i+1,j+1,k+1] = 2*ηc[i,j,k]*εzz[i,j,k]
    end

    @inbounds if i<size(εxx,1)+1 && j<size(εxx,2)+1 && k<size(εxx,3)+1
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
    return
end

# ###############################

@tiny function Kernel_ComputeResiduals!( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, P0, K, ∇V, Δx, Δy, Δz, Δt )
    _Δx, _Δy, _Δz, _KΔt = 1.0/Δx, 1.0/Δy, 1.0/Δz, 1.0/K/Δt
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
            Fp[i,j,k] = -∇V[i,j,k] - (P[i+1,j+1,k+1] - P0[i+1,j+1,k+1])*_KΔt #-P0[i,j] 
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
            Vx[i,j+1,k+1] += hx[i,j,k]*dVxdτ[i,j,k]
        end

        if (i<=size(dVydτ,1)) && (j<=size(dVydτ,2)) && (k<=size(dVydτ,3))
            Vy[i+1,j,k+1] += hy[i,j,k]*dVydτ[i,j,k]
        end

        if (i<=size(dVzdτ,1)) && (j<=size(dVzdτ,2)) && (k<=size(dVzdτ,3))
            Vz[i+1,j+1,k] += hz[i,j,k]*dVzdτ[i,j,k]
        end

        if (i<=size(dPdτ,1)) && (j<=size(dPdτ,2)) && (k<=size(dPdτ,3))
            P[i+1,j+1,k+1] += hp[i,j,k]*dPdτ[i,j,k] 
        end
    end
    return
end

#############################################################################################
#############################################################################################
#############################################################################################

function Stokes3D(n, ::Type{DAT}; device) where DAT
    Lx,  Ly,  Lz  =  2.0,  1.0,  1.0 
    ncx, ncy, ncz = n*32*2, n*32, n*32
    BCtype = :PureShear_xz
    ε      = 1
    r      = 0.25
    #-----------
    P     = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(P,  DAT(0.))
    P0    = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(P0, DAT(0.))
    Vx    = device_array(DAT, device, ncx+1, ncy+2, ncz+2); fill!(Vx, DAT(0.))
    Vy    = device_array(DAT, device, ncx+2, ncy+1, ncz+2); fill!(Vy, DAT(0.))
    Vz    = device_array(DAT, device, ncx+2, ncy+2, ncz+1); fill!(Vz, DAT(0.))
    ηc    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(ηc, DAT(0.))
    ηv    = device_array(DAT, device, ncx+1, ncy+1, ncz+1); fill!(ηv, DAT(0.))
    τxx   = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τxx, DAT(0.))
    τyy   = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τyy, DAT(0.))
    τzz   = device_array(DAT, device, ncx+2, ncy+2, ncz+2); fill!(τzz, DAT(0.))
    τxy   = device_array(DAT, device, ncx+1, ncy+1, ncz+0); fill!(τxy, DAT(0.))
    τxz   = device_array(DAT, device, ncx+1, ncy+0, ncz+1); fill!(τxz, DAT(0.))
    τyz   = device_array(DAT, device, ncx+0, ncy+1, ncz+1); fill!(τyz, DAT(0.))
    ηxy   = device_array(DAT, device, ncx+1, ncy+1, ncz+0); fill!(ηxy, DAT(0.))
    ηxz   = device_array(DAT, device, ncx+1, ncy+0, ncz+1); fill!(ηxz, DAT(0.))
    ηyz   = device_array(DAT, device, ncx+0, ncy+1, ncz+1); fill!(ηyz, DAT(0.))
    ∇V    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(∇V,  DAT(0.))
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
    hx    = device_array(DAT, device, ncx+1, ncy+0, ncz+0); fill!(hx,  DAT(0.))
    hy    = device_array(DAT, device, ncx+0, ncy+1, ncz+0); fill!(hy,  DAT(0.))
    hz    = device_array(DAT, device, ncx+0, ncy+0, ncz+1); fill!(hz,  DAT(0.))
    hp    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(hp,  DAT(0.))
    λx    = device_array(DAT, device, ncx+1, ncy+0, ncz+0); fill!(λx,  DAT(0.))
    λy    = device_array(DAT, device, ncx+0, ncy+1, ncz+0); fill!(λy,  DAT(0.))
    λz    = device_array(DAT, device, ncx+0, ncy+0, ncz+1); fill!(λz,  DAT(0.))
    λp    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(λp,  DAT(0.))

    δx    = device_array(DAT, device, ncx+1, ncy+0, ncz+0); fill!(δx,  DAT(0.))
    δy    = device_array(DAT, device, ncx+0, ncy+1, ncz+0); fill!(δy,  DAT(0.))
    δz    = device_array(DAT, device, ncx+0, ncy+0, ncz+1); fill!(δz,  DAT(0.))
    δp    = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(δp,  DAT(0.))

    #-----------
    xv = LinRange(-Lx/2, Lx/2, ncx+1)
    yv = LinRange(-Ly/2, Ly/2, ncy+1)
    zv = LinRange(-Lz/2, Lz/2, ncz+1)
    Δx, Δy, Δz = Lx/ncx, Ly/ncy, Lz/ncz
    xce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2)
    yce = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    zce = LinRange(-Lz/2-Δz/2, Lz/2+Δz/2, ncz+2)
    Δt  = 1.0
    K   = 1.0

    # Kernels
    InitialCondition!   = Kernel_InitialCondition!(device)
    SetBoundaries!      = Kernel_SetBoundaries!(device)
    InterpV2C!          = Kernel_InterpV2C!(device)
    InterpV2xyz!        = Kernel_InterpV2xyz!(device)  
    ComputeStrainRates! = Kernel_ComputeStrainRates!(device)
    ComputeStress!      = Kernel_ComputeStress!(device)
    ComputeResiduals!   = Kernel_ComputeResiduals!(device)
    UpdateRates!        = Kernel_UpdateRates!(device)
    UpdateVP!           = Kernel_UpdateVP!(device)
    Gerschgorin1!       = Kernel_Gerschgorin1!(device)
    Gerschgorin2!       = Kernel_Gerschgorin2!(device)
    Gerschgorin3!       = Kernel_Gerschgorin3!(device)
    MaxLoc!             = Kernel_MaxLoc!(device)
    TinyKernels.device_synchronize(device)
    
    wait( InitialCondition!( Vx, Vy, Vz, ηv, ηc, ε, xv, yv, zv, xce, yce, zce, r; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    wait( InterpV2C!( ηc, ηv; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    wait( InterpV2xyz!( ηxy, ηxz, ηyz, ηv; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    ηv2   = device_array(DAT, device, ncx+1, ncy+1, ncz+1); fill!(ηv2, DAT(0.))
    ηv2  .= ηv
    wait( MaxLoc!(ηv2, ηv; ndrange=(ncx+2, ncy+2, ncz+2)) )

    ηc2   = device_array(DAT, device, ncx+0, ncy+0, ncz+0); fill!(ηc2, DAT(0.))
    ηc2  .= ηc
    wait( MaxLoc!(ηc2, ηc; ndrange=(ncx+2, ncy+2, ncz+2)) )

    # --------- DYREL --------- #
    niter  = 1e5
    nout   = 2000
    tol    = 1e-6

    # Steps
    hx    .= 1.0 ./av2yz_arit(ηv2)*4
    hy    .= 1.0 ./av2zx_arit(ηv2)*4
    hz    .= 1.0 ./av2xy_arit(ηv2)*4
    hp    .= 1.0  

    # Gerschgorin
    wait( Gerschgorin1!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Δx, Δy, Δz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    wait( Gerschgorin2!( τxx, τyy, τzz, τxy, τxz, τyz, εxx, εyy, εzz, εxy, εxz, εyz, ηc, ηxy, ηxz, ηyz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    wait( Gerschgorin3!( λx, λy, λz, λp, τxx, τyy, τzz, τxy, τxz, τyz, P, P0, K, ∇V, Δx, Δy, Δz, Δt; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    @show λmax =  maximum([maximum(λx.*hx) maximum(λy.*hy) maximum(λz.*hz) maximum(λp.*hp)])
    minimum([minimum(λx[2:end-1,:,:].*hx[2:end-1,:,:]) minimum(λy[:,2:end-1,:].*hy[:,2:end-1,:]) minimum(λz[:,:,2:end-1].*hz[:,:,2:end-1]) minimum(λp.*hp)])

    @show λmin = 1.0
    h_ρ   = 4.0/(λmin + λmax)
    ch_ρ  = 4.0*sqrt(λmin*λmax)/(λmin + λmax)

    # --------- DYREL --------- #

    for iter=1:niter
        wait( SetBoundaries!( Vx, Vy, Vz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
        wait( ComputeStrainRates!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Vx, Vy, Vz, Δx, Δy, Δz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
        wait( ComputeStress!( τxx, τyy, τzz, τxy, τxz, τyz, εxx, εyy, εzz, εxy, εxz, εyz, ηc, ηxy, ηxz, ηyz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
        wait( ComputeResiduals!( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, P0, K, ∇V, Δx, Δy, Δz, Δt; ndrange=(ncx+2, ncy+2, ncz+2) ) )  
        wait( UpdateRates!( dVxdτ, dVydτ, dVzdτ, dPdτ, Fx, Fy, Fz, Fp, ch_ρ, h_ρ, ncx, ncy, ncz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
        wait( UpdateVP!( dVxdτ, dVydτ, dVzdτ, dPdτ, Vx, Vy, Vz, P, hx, hy, hz, hp, ncx, ncy, ncz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
    
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
        # if mod(iter,nout) == 0 
        #     # Joldes et al. (2011) --- v4 
        #     δx   .= hx.*dVxdτ
        #     δy   .= hy.*dVydτ
        #     δz   .= hz.*dVzdτ
        #     δp   .= hp.*dPdτ
        #     wait( SetBoundaries!( δx, δy, δz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
        #     wait( ComputeStrainRates!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, -hx.*δx, -hy.*δy, -hz.*δz, Δx, Δy, Δz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
        #     wait( ComputeStress!( τxx, τyy, τzz, τxy, τxz, τyz, εxx, εyy, εzz, εxy, εxz, εyz, ηc, ηxy, ηxz, ηyz; ndrange=(ncx+2, ncy+2, ncz+2) ) )
        #     wait( ComputeResiduals!( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, -hp.*δp, P0, K, ∇V, Δx, Δy, Δz, Δt; ndrange=(ncx+2, ncy+2, ncz+2) ) )
        #     λmin  = 2*(abs(sum(δx.*Fx)) + abs(sum(δy.*Fy)) + abs(sum(δz.*Fz)) + abs(sum(δp.*Fp))) / (sum(δx.*δx./hx./hx) + sum(δy.*δy./hy./hy) + sum(δz.*δz./hz./hz) + sum(δp.*δp./hp./hp))
        #     @show λmin
        #     h_ρ   = 4.0/(λmin + λmax)
        #     ch_ρ  = 4.0*sqrt(λmin*λmax)/(λmin + λmax)
        # end
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
    end
    
    #-----------
    return nothing
end

#############################################################################################
#############################################################################################
#############################################################################################

@time Stokes3D( 10, eletype; device )
