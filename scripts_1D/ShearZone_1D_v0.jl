using Plots
ymin = -1.0
ymax =  1.0
ncy  = 10
Δy   = (ymax-ymin)/ncy
yv   = LinRange(     ymin,      ymax, ncy+1)
yc   = LinRange(ymin+Δy/2, ymax-Δy/2, ncy  )

NumVx = (1:(ncy+1))
NumVy = (1:(ncy+1)) .+ maximum(NumVx)
NumPt = 1:ncy
