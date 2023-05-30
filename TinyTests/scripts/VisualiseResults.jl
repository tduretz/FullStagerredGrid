using GLMakie

function PlotResults()
    x = [    3.8e15;  8.9e15;  1.073e16]
    n = [1.; 2.; 4.]
    lines(log10.(n), log10.(x) )
end
PlotResults()