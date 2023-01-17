# This message reports the velocity in local North East Down (NED) coordinates.
# The NED coordinate system is defined as the local WGS84 tangent plane centered
# at the current position. The full GPS time is given by the preceding MSG GPS
# TIME with the matching time-of-week (tow). This message is similar to the MSG
# VEL NED, but it includes the upper triangular portion of the 3x3 covariance
# matrix.

uint32 tow          # GPS Time of Week.
int32 n             # Velocity North coordinate [mm/s].
int32 e             # Velocity East coordinate [mm/s].
int32 d             # Velocity Down coordinate [mm/s].
float32 cov_n_n     # Estimated variance of northward measurement [m^2/s^2].
float32 cov_n_e     # Covariance of northward and eastward measurement [m^2/s^2].
float32 cov_n_d     # Covariance of northward and downward measurement [m^2/s^2].
float32 cov_e_e     # Estimated variance of eastward measurement [m^2/s^2].
float32 cov_e_d     # Covariance of eastward and downward measurement [m^2/s^2].
float32 cov_d_d     # Estimated variance of downard measurement [m^2/s^2].
uint8 n_sats        # Number of satellites used in solution.
uint8 flags         # Status flags

uint8 VEL_MODE_INVALID = 0
uint8 VEL_MODE_MEAS_DOPPLER = 1
uint8 VEL_MODE_DERIVED_DOPPLER = 2
uint8 VEL_MODE_DEAD_RECKONING = 3
