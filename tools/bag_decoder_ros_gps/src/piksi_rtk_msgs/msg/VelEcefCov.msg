# This message reports the velocity in local North East Down (NED) coordinates.
# The NED coordinate system is defined as the local WGS84 tangent plane centered
# at the current position. The full GPS time is given by the preceding MSG GPS
# TIME with the matching time-of-week (tow). This message is similar to the MSG
# VEL NED, but it includes the upper triangular portion of the 3x3 covariance
# matrix.

uint32 tow          # GPS Time of Week.
int32 x             # Velocity North coordinate [mm/s].
int32 y             # Velocity East coordinate [mm/s].
int32 z             # Velocity Down coordinate [mm/s].
float32 cov_x_x     # Estimated variance of x [m^2/s^2].
float32 cov_x_y     # Estimated covariance of x and y [m^2/s^2].
float32 cov_x_z     # Estimated covariance of x and z [m^2/s^2].
float32 cov_y_y     # Estimated variance of y [m^2/s^2].
float32 cov_y_z     # Estimated covariance of y and z [m^2/s^2].
float32 cov_z_z     # Estimated variance of z [m^2/s^2].
uint8 n_sats        # Number of satellites used in solution.
uint8 flags         # Status flags

uint8 VEL_MODE_INVALID = 0
uint8 VEL_MODE_MEAS_DOPPLER = 1
uint8 VEL_MODE_DERIVED_DOPPLER = 2
uint8 VEL_MODE_DEAD_RECKONING = 3
