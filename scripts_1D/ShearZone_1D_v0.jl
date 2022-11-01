using Plots
ymin = -1.0
ymax =  1.0
ε̇bg  =  1.0

ncy  = 10
Δy   = (ymax-ymin)/ncy
yv   = LinRange(     ymin,      ymax, ncy+1)
yc   = LinRange(ymin+Δy/2, ymax-Δy/2, ncy  )

NumVx = (1:(ncy+1))
NumVy = (1:(ncy+1)) .+ maximum(NumVx)
NumPt = 1:ncy

Vx    = yv.*ε̇bg
Vy    = 0.0.*yv
Pt    = 0.0.*yc

bct   = zeros(Int,ncy+1)
bct[1] = 1; bct[end] = 1

I = ones(Int,3*2*(ncy+1))
J = ones(Int,3*2*(ncy+1))
V = zeros(3*2*(ncy+1))

# Construct velocity block
for idof in eachindex(Vx)

    if bct[idof] == 1
    else
    end
end

