# This message reports the velocity in local North East Down (NED) coordinates. The NED coordinate
# system is defined as the local WGS84 tangent plane centered at the current position. The full GPS time
# is given by the preceding MSG GPS TIME with the matching time-of-week (tow).

Header header

uint32 tow          # GPS Time of Week.
int32 n             # Velocity North coordinate [mm/s].
int32 e             # Velocity East coordinate [mm/s].
int32 d             # Velocity Down coordinate [mm/s].
uint16 h_accuracy   # Horizontal velocity accuracy estimate (not implemented). Defaults to 0.
uint16 v_accuracy   # Vertical velocity accuracy estimate (not implemented). Defaults to 0.
uint8 n_sats        # Number of satellites used in solution.
uint8 flags         # Status flags (reserved), see MSG_VEL_ECEF message description in SBP documentation.
