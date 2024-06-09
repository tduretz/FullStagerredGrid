@tiny function Kernel_Gerschgorin1!( ∇V, εxx, εyy, εzz, εxy, εxz, εyz, Δx, Δy, Δz )
    _Δx, _Δy, _Δz = 1.0/Δx, 1.0/Δy, 1.0/Δz
    i, j, k = @indices
     if (i<=size(εxx,1)+0) && (j<=size(εxx,2)+0) && (k<=size(εxx,3)+0)
        dVxΔx      = 2.0 * _Δx
        dVyΔy      = 2.0 * _Δy
        dVzΔz      = 2.0 * _Δz
        ∇V[i,j,k]  = dVxΔx + dVyΔy + dVzΔz
        εxx[i,j,k] = dVxΔx - 1.0/3.0 * ∇V[i,j,k]
        εyy[i,j,k] = dVyΔy - 1.0/3.0 * ∇V[i,j,k]
        εzz[i,j,k] = dVzΔz - 1.0/3.0 * ∇V[i,j,k]
    end

     if i<size(εxx,1)+1 && j<size(εxx,2)+1 && k<size(εxx,3)+1
        if (i<=size(εxy,1)) && (j<=size(εxy,2)) && (k<=size(εxy,3))
            dVxΔy      = 2.0 *_Δy 
            dVyΔx      = 2.0 *_Δx 
            εxy[i,j,k] = 0.5*(dVxΔy + dVyΔx)
        end
        if (i<=size(εxz,1)) && (j<=size(εxz,2)) && (k<=size(εxz,3))
            dVxΔz      = 2.0 *_Δz                     
            dVzΔx      = 2.0 *_Δx 
            εxz[i,j,k] = 0.5*(dVxΔz + dVzΔx)
        end
        if (i<=size(εyz,1)) && (j<=size(εyz,2)) && (k<=size(εyz,3))
            dVyΔz      = 2.0 *_Δz 
            dVzΔy      = 2.0 *_Δy 
            εyz[i,j,k] = 0.5*(dVyΔz + dVzΔy)
        end
    end
    return
end

# ###############################

@tiny function Kernel_Gerschgorin2!( τxx, τyy, τzz, τxyc, τxzc, τyzc, τxx0, τyy0, τzz0, τxy0, τxz0, τyz0, εxx, εyy, εzz, εxy, εxz, εyz, ηc, Gc, Gv, P, Δt, pl )
    i, j, k = @indices
    @inline isin(A) = checkbounds(Bool, A, i, j, k)
     if isin(εxx)
        ηeff = ηc[i,j,k]
        ηe   = Gc[i,j,k]*Δt
        εxxc = εxx[i,j,k] + τxx0[i+1,j+1,k+1]/2ηe
        εyyc = εyy[i,j,k] + τyy0[i+1,j+1,k+1]/2ηe
        εzzc = εzz[i,j,k] + τzz0[i+1,j+1,k+1]/2ηe
        εxyc = 0.25*( (εxy[i,j,k] + τxy0[i,j,k]/2/Δt/Gv[i,j,k]) + (εxy[i+1,j,k] + τxy0[i+1,j,k]/2/Δt/Gv[i+1,j,k]) + (εxy[i,j+1,k] + τxy0[i,j+1,k]/2/Δt/Gv[i,j+1,k]) + (εxy[i+1,j+1,k] + τxy0[i+1,j+1,k]/2/Δt/Gv[i+1,j+1,k]))
        εxzc = 0.25*( (εxz[i,j,k] + τxz0[i,j,k]/2/Δt/Gv[i,j,k]) + (εxz[i+1,j,k] + τxz0[i+1,j,k]/2/Δt/Gv[i+1,j,k]) + (εxz[i,j,k+1] + τxz0[i,j,k+1]/2/Δt/Gv[i,j,k+1]) + (εxz[i+1,j,k+1] + τxz0[i+1,j,k+1]/2/Δt/Gv[i+1,j,k+1]))
        εyzc = 0.25*( (εyz[i,j,k] + τyz0[i,j,k]/2/Δt/Gv[i,j,k]) + (εyz[i,j+1,k] + τyz0[i,j+1,k]/2/Δt/Gv[i,j+1,k]) + (εyz[i,j,k+1] + τyz0[i,j,k+1]/2/Δt/Gv[i,j,k+1]) + (εyz[i,j+1,k+1] + τyz0[i,j+1,k+1]/2/Δt/Gv[i,j+1,k+1]))
        τxx[i+1,j+1,k+1]  = 2*ηeff*εxxc
        τyy[i+1,j+1,k+1]  = 2*ηeff*εyyc
        τzz[i+1,j+1,k+1]  = 2*ηeff*εzzc
        τxyc[i+1,j+1,k+1] = 2*ηeff*εxyc
        τxzc[i+1,j+1,k+1] = 2*ηeff*εxzc
        τyzc[i+1,j+1,k+1] = 2*ηeff*εyzc
    end
    return
end

@tiny function Kernel_Gerschgorin2b!( τxyc, τxzc, τyzc, τxy, τxz, τyz)
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

@tiny function Kernel_Gerschgorin3!( Fx, Fy, Fz, Fp, τxx, τyy, τzz, τxy, τxz, τyz, P, P0, K, ∇V, Δx, Δy, Δz, Δt )
    _Δx, _Δy, _Δz, _KΔt = 1.0/Δx, 1.0/Δy, 1.0/Δz, 1.0/K/Δt
    i, j, k = @indices
     if i<size(P,1)-1 && j<size(P,2)-1 && k<size(P,3)-1
        if (i<=size(Fx,1)) && (j<=size(Fx,2)) && (k<=size(Fx,3))
            if (i>1 && i<size(Fx,1)) # avoid Dirichlets
                Fx[i,j,k]  = (τxx[i+1,j+1,k+1] + τxx[i,j+1,k+1]) *_Δx
                Fx[i,j,k] += (  P[i+1,j+1,k+1] +   P[i,j+1,k+1]) *_Δx
                Fx[i,j,k] += (τxy[i,j+1,k] + τxy[i,j,k]) *_Δy
                Fx[i,j,k] += (τxz[i,j,k+1] + τxz[i,j,k]) *_Δz
            end
        end
        if (i<=size(Fy,1)) && (j<=size(Fy,2)) && (k<=size(Fy,3))
            if (j>1 && j<size(Fy,2)) # avoid Dirichlets
                Fy[i,j,k]  = (τyy[i+1,j+1,k+1] + τyy[i+1,j,k+1]) *_Δy
                Fy[i,j,k] += (  P[i+1,j+1,k+1] +   P[i+1,j,k+1]) *_Δy
                Fy[i,j,k] += (τxy[i+1,j,k] + τxy[i,j,k]) *_Δx
                Fy[i,j,k] += (τyz[i,j,k+1] + τyz[i,j,k]) *_Δz
            end
        end
        if (i<=size(Fz,1)) && (j<=size(Fz,2)) && (k<=size(Fz,3))
            if (k>1 && k<size(Fz,3)) # avoid Dirichlets
                Fz[i,j,k]  = (τzz[i+1,j+1,k+1] + τzz[i+1,j+1,k]) *_Δz
                Fz[i,j,k] += (  P[i+1,j+1,k+1] +   P[i+1,j+1,k]) *_Δz
                Fz[i,j,k] += (τxz[i+1,j,k] + τxz[i,j,k]) *_Δx
                Fz[i,j,k] += (τyz[i,j+1,k] + τyz[i,j,k]) *_Δy
            end
        end
        if (i<=size(Fp,1)) && (j<=size(Fp,2)) && (k<=size(Fp,3))
            Fp[i,j,k] = ∇V[i,j,k] + 1.0*_KΔt 
        end
    end
    return
end