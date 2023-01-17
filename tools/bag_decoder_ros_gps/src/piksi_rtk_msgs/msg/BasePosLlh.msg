# The base station position message is the position reported by the base station itself.
# It is used for pseudo-absolute RTK positioning, and is required to be a high-accuracy surveyed location of
# the base station. Any error here will result in an error in the pseudo-absolute position output.

Header header

float64 lat             # Latitude [deg].
float64 lon             # Longitude [deg].
float64 height          # Height [m].
