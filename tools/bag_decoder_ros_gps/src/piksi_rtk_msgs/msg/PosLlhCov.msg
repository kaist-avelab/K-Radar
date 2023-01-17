# This position solution message reports the absolute geodetic coordinates and
# the status (single point vs pseudo-absolute RTK) of the position solution as
# well as the upper triangle of the 3x3 covariance matrix. The position
# information and Fix Mode flags should follow the MSG POS LLH message.
# Since the covariance matrix is computed in the local-level North, East, Down
# frame, the covariance terms follow with that convention. Thus, covariances are
# reported against the ”downward” measurement and care should be taken with the
# sign convention.

Header header

uint32 tow          # GPS Time of Week [ms].
float64 lat         # Latitude [deg].
float64 lon         # Longitude [deg].
float64 height      # Height [deg].
float32 cov_n_n     # Estimated variance of northing [m^2].
float32 cov_n_e     # Covariance of northing and easting [m^2].
float32 cov_n_d     # Covariance of northing and downward measurement [m^2].
float32 cov_e_e     # Estimated variance of easting [m^2].
float32 cov_e_d     # Covariance of easting and downward measurement [m^2].
float32 cov_d_d     # Estimated variance of downward measurement [m^2].
uint8 n_sats        # Number of satellites used in solution.
uint8 flags         # Status flags, see MSG_POS_LLH message description in SBP documentation.

uint8 FIX_MODE_INVALID = 0
uint8 FIX_MODE_SPP = 1
uint8 FIX_MODE_DGNSS = 2
uint8 FIX_MODE_FLOAT_RTK = 3
uint8 FIX_MODE_FIX_RTK = 4
uint8 FIX_MODE_DEAD_RECKONING = 5
uint8 FIX_MODE_SBAS = 6
