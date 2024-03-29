using Makie

function PatchPlotMakieBasic(vertx, verty, sol, xmin, xmax, ymin, ymax; cmap = :turbo, write_fig=false )
    
    Makie.inline!(false)
    f   = Figure(resolution = (1200, 1000))

    ar = (xmax - xmin) / (ymax - ymin)

    Axis(f[1,1]) #, aspect = ar
    min_v = minimum( sol.p ); max_v = maximum( sol.p )

    # min_v = .0; max_v = 5.

    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.p)]
    # poly!(p, color = sol.p, colormap = cmap, strokewidth = 1, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # Axis(f[2,1], aspect = ar)
    min_v = minimum( sol.vx ); max_v = maximum( sol.vx )
    limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    poly!(p, color = sol.vx, colormap = cmap, strokewidth = 0.5, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)

    # Axis(f[2,2], aspect = ar)
    # min_v = minimum( sol.vy ); max_v = maximum( sol.vy )
    # limits = min_v ≈ max_v ? (min_v, min_v + 1) : (min_v, max_v)
    # p = [Polygon( Point2f0[ (vertx[i,j], verty[i,j]) for j=1:4] ) for i in 1:length(sol.vx)]
    # poly!(p, color = sol.vy, colormap = cmap, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=limits)
    
    # scatter!(x1,y1, color=:white)
    # scatter!(x2,y2, color=:white, marker=:xcross)
    Colorbar(f[1, 2], colormap = cmap, limits=limits, flipaxis = true, size = 25 )

    
    display(f)
    if write_fig==true 
        FileIO.save( string(@__DIR__, "/plot.png"), f)
    end
    return nothing
end