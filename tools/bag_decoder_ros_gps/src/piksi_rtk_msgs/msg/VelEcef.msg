# This message reports the velocity in Earth Centered Earth Fixed (ECEF) coordinates. The full GPS time
# is given by the preceding MSG GPS TIME with the matching time-of-week (tow).

Header header

uint32 tow      # GPS Time of Week.
int32 x         # Velocity ECEF X coordinate [mm/s].
int32 y         # Velocity ECEF Y coordinate [mm/s].
int32 z         # Velocity ECEF Z coordinate [mm/s].
uint16 accuracy # Velocity accuracy estimate (not implemented). Defaults to 0.
uint8 n_sats    # Number of satellites used in solution.
uint8 flags     # Status flags (reserved), see MSG_VEL_ECEF message description in SBP documentation.
