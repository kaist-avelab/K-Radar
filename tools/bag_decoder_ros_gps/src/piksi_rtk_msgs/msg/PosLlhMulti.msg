# This position solution message reports the absolute geodetic coordinates and the status (single point
# vs pseudo-absolute RTK) of the position solution. If the rover receiver knows the surveyed position of
# the base station and has an RTK solution, this reports a pseudo-absolute position solution using the
# base station position and the rover's RTK baseline vector. The full GPS time is given by the preceding
# MSG GPS TIME with the matching time-of-week (tow).

Header header

uint32 tow          # GPS Time of Week [ms].
float64 lat         # Latitude [deg].
float64 lon         # Longitude [deg].
float64 height      # Height [deg].
uint16 h_accuracy   # Horizontal position accuracy estimate.
uint16 v_accuracy   # Vertical position accuracy estimate.
uint8 n_sats        # Number of satellites used in solution.
uint8 flags         # Status flags, see MSG_POS_LLH message description in SBP documentation.

uint8 FIX_MODE_INVALID = 0
uint8 FIX_MODE_SPP = 1
uint8 FIX_MODE_DGNSS = 2
uint8 FIX_MODE_FLOAT_RTK = 3
uint8 FIX_MODE_FIX_RTK = 4
uint8 FIX_MODE_DEAD_RECKONING = 5
uint8 FIX_MODE_SBAS = 6
