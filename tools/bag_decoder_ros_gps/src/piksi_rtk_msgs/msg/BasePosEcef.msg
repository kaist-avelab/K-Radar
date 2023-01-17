# The base station position message is the position reported by the base station itself in absolute Earth
# Centered Earth Fixed coordinates. It is used for pseudo-absolute RTK positioning, and is required to
# be a high-accuracy surveyed location of the base station. Any error here will result in an error in the
# pseudo-absolute position output.

Header header

float64 x             # ECEF X coodinate [m].
float64 y             # ECEF Y coodinate [m].
float64 z             # ECEF Z coodinate [m].
