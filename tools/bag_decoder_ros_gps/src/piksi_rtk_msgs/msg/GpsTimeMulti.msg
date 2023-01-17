# This message reports the GPS time, representing the time since the GPS epoch began on midnight
# January 6, 1980 UTC. GPS time counts the weeks and seconds of the week. The weeks begin at the
# Saturday/Sunday transition. GPS week 0 began at the beginning of the GPS time scale.
# Within each week number, the GPS time of the week is between between 0 and 604800 seconds
# (=60*60*24*7). Note that GPS time does not accumulate leap seconds, and as of now, has a small offset
# from UTC. In a message stream, this message precedes a set of other navigation messages referenced
# to the same time (but lacking the ns field) and indicates a more precise time of these messages.

Header header

uint32 wn          # GPS week number [weeks].
uint32 tow         # GPS time of week rounded to the nearest millisecond [ms].
int32 ns_residual  # Nanosecond residual of millisecond-rounded TOW (ranges from -500000 to 500000) [ns].
uint8 flags        # Status flags (reserved).
