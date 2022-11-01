using Plots, Printf, SparseArrays, LinearAlgebra

function UpdateCOO!(I,J,V, i,j,v, innz)
    I[innz] = i
    J[innz] = j
    V[innz] = v
    return innz   += 1 
end

function Main_1D()
    ymin = -1.0
    ymax =  1.0
    ε̇bg  =  1.0

    ncy  = 10#10000000
    Δy   = (ymax-ymin)/ncy
    yv   = LinRange(     ymin,      ymax, ncy+1)
    yc   = LinRange(ymin+Δy/2, ymax-Δy/2, ncy  )
    dy = Δy

    NumVx = (1:(ncy+1))
    NumVy = (1:(ncy+1)) .+ maximum(NumVx)
    NumPt = 1:ncy
    η     = ones(ncy)

    Vx    = yv.*ε̇bg
    Vy    = 0.0.*yv
    Pt    = 0.0.*yc

    bct   = zeros(Int,ncy+1)
    bct[1] = 1; bct[end] = 1

    I = ones(Int,3*2*(ncy+1))
    J = ones(Int,3*2*(ncy+1))
    V = zeros(3*2*(ncy+1))

    # Construct velocity block
    innz = 1
    @time for inode in eachindex(Vx)
        iVxC = NumVx[inode] 
        iVyC = NumVy[inode] 
        # Connectivity
        if bct[inode] == 1
            # Vx Dirichlet
            innz = UpdateCOO!(I,J,V, iVxC,iVxC,1.0, innz)
            # Vy Dirichlet
            innz = UpdateCOO!(I,J,V, iVyC,iVyC,1.0, innz)
        else
            eS   = η[inode-1]
            eN   = η[inode]
            iVxS = NumVx[inode-1] 
            iVxN = NumVx[inode+1] 
            iVyS = NumVy[inode-1] 
            iVyN = NumVy[inode+1] 
            cVxC = (1.0 * eN ./ dy + 1.0 * eS ./ dy) ./ dy
            cVxS = -1.0 * eS ./ dy .^ 2
            cVxN = -1.0 * eN ./ dy .^ 2
            cVyC = ((4 // 3) * eN ./ dy + (4 // 3) * eS ./ dy) ./ dy
            cVyS = -4 // 3 * eS ./ dy .^ 2
            cVyN = -4 // 3 * eN ./ dy .^ 2
            if bct[inode-1] == 0
                innz = UpdateCOO!(I,J,V, iVxC,iVxS,cVxS, innz)
                innz = UpdateCOO!(I,J,V, iVyC,iVyS,cVyS, innz)
            end
            innz = UpdateCOO!(I,J,V, iVxC,iVxC,cVxC, innz)
            innz = UpdateCOO!(I,J,V, iVyC,iVyC,cVyC, innz)
            @show iVyC
            @show cVyC
            if bct[inode+1] == 0
                innz = UpdateCOO!(I,J,V, iVxC,iVxN,cVxN, innz)
                innz = UpdateCOO!(I,J,V, iVyC,iVyN,cVyN, innz)
            end
        end

    end
    @printf("Number of dofs: %06d\n", innz)
    Kuu = sparse(I, J, V)
    display(spy(Kuu))
    # Kc = cholesky(Kuu)
end 

Main_1D()

