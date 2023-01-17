# This message reports the baseline heading pointing from the base station to the rover relative to True
# North. The full GPS time is given by the preceding MSG GPS TIME with the matching time-of-week
# (tow). It is intended that time-matched RTK mode is used when the base station is moving.

Header header

uint32 tow          # GPS Time of Week [ms].
uint32 heading      # Heading [mdeg].
uint8  n_sats       # Number of satellites used in solution
uint8  flags        # Status flag.

