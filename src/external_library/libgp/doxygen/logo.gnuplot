set term postscript

set out "logo.ps"

unset hidden3d

# draw the surface using pm3d's hidden3d with line type 100
unset hidden
unset surface
unset key

set samples 20; set isosamples 20
set palette defined (0 0.23 0.33 0.53, 1 0.28 0.57 0.81)
set pm3d
set style line 100 lt rgb "#F0EAD6" lw 0.4
set pm3d hidden3d 100
set xrange [-4:4]
set yrange [-4:4]

unset tics
unset border
unset colorbox

splot exp(-(x*x/15+y*y/15)) 

