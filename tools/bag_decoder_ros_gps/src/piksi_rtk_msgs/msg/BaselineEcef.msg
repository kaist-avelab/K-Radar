# This message reports the baseline solution in Earth Centered Earth Fixed (ECEF) coordinates.
# This baseline is the relative vector distance from the base station to the rover receiver.
# The full GPS time is given by the preceding MSG GPS TIME with the matching time-of-week (tow).

Header header

uint32 tow          # GPS Time of Week [ms].
int32 x             # Baseline ECEF X coordinate [mm].
int32 y             # Baseline ECEF Y coordinate [mm].
int32 z             # Baseline ECEF Z coordinate [mm].
uint16 accuracy     # Position accuracy estimate (not implemented in hardware version V2).
uint8 n_sats        # Number of satellites used in solution.
uint8 flags         # Status flags, see MSG_BASELINE_ECEF message description in SBP documentation.
