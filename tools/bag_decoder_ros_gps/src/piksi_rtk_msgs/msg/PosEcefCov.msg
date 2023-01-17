# The position solution message reports absolute Earth Centered Earth Fixed
# (ECEF) coordinates and the status (single point vs pseudo-absolute RTK) of the
# position solution. The message also reports the upper triangular portion of
# the 3x3 covariance matrix. If the receiver knows the surveyed position of the
# base station and has an RTK solution, this reports a pseudo-absolute position
# solution using the base station position and the rover’s RTK baseline vector.
# The full GPS time is given by the preceding MSG GPS TIME with the matching
# time-of-week (tow).

Header header

uint32 tow          # GPS Time of Week [ms].
float64 x           # ECEF X coordinate [m].
float64 y           # ECEF Y coordinate [m].
float64 z           # ECEF Z coordinate [m].
float32 cov_x_x     # Estimated variance of x [m^2].
float32 cov_x_y     # Estimated covariance of x and y [m^2].
float32 cov_x_z     # Estimated covariance of x and z [m^2].
float32 cov_y_y     # Estimated variance of y [m^2].
float32 cov_y_z     # Estimated covariance of y and z [m^2].
float32 cov_z_z     # Estimated variance of z [m^2].
uint8 n_sats        # Number of satellites used in solution.
uint8 flags         # Status flags, see MSG_POS_ECEF message description in SBP documentation.

uint8 FIX_MODE_INVALID = 0
uint8 FIX_MODE_SPP = 1
uint8 FIX_MODE_DGNSS = 2
uint8 FIX_MODE_FLOAT_RTK = 3
uint8 FIX_MODE_FIX_RTK = 4
uint8 FIX_MODE_DEAD_RECKONING = 5
uint8 FIX_MODE_SBAS = 6
