# This message reports the baseline solution in North East Down (NED) coordinates. This baseline is
# the relative vector distance from the base station to the rover receiver, and NED coordinate system is
# defined at the local WGS84 tangent plane centered at the base station position. The full GPS time is
# given by the preceding MSG GPS TIME with the matching time-of-week (tow).

Header header

uint32 tow          # GPS Time of Week [ms].
int32 n             # Baseline North coordinate [mm].
int32 e             # Baseline East coordinate[mm].
int32 d             # Baseline Down coordinate [mm].
uint16 h_accuracy   # Horizontal position accuracy estimate (not implemented in hardware version V2).
uint16 v_accuracy   # Vertical position accuracy estimate (not implemented in hardware version V2).
uint8 n_sats        # Number of satellites used in solution
uint8 flags         # Status flags, see MSG_BASELINE_NED message description in SBP documentation.
