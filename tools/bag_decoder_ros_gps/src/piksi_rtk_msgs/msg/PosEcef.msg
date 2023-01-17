# The position solution message reports absolute Earth Centered Earth Fixed (ECEF) coordinates and the
# status (single point vs pseudo-absolute RTK) of the position solution. If the rover receiver knows the
# surveyed position of the base station and has an RTK solution, this reports a pseudo-absolute position
# solution using the base station position and the rover's RTK baseline vector. The full GPS time is given
# by the preceding MSG GPS TIME with the matching time-of-week (tow).

Header header

uint32 tow          # GPS Time of Week [ms].
float64 x           # ECEF X coordinate [m].
float64 y           # ECEF Y coordinate [m].
float64 z           # ECEF Z coordinate [m].
uint16 accuracy     # Position accuracy estimate (not implemented in hardware version V2).
uint8 n_sats        # Number of satellites used in solution.
uint8 flags         # Status flags, see MSG_POS_ECEF message description in SBP documentation.
