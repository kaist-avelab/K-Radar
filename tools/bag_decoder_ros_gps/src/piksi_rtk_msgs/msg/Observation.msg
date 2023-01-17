# Satellite observation messages from the device.

Header header

uint32 tow                  # Milliseconds since start of GPS week.
int32  ns_residual          # Nanosecond residual of millisecond-rounded TOW (ranges from -500000 to 500000).
uint16 wn                   # GPS week number.
uint8  n_obs                # Total number of observations.
                            # First nibble is the size of the sequence (n), second nibble is the zero-indexed counter (ith packet of n).

uint32[] P                  # Pseudorange observation.
int32[]  L_i                # Carrier phase whole cycles.
uint8[]  L_f                # Carrier phase fractional part.
int16[]  D_i                # Doppler whole Hz.
uint8[]  D_f                # Doppler fractional part.
uint8[]  cn0                # Carrier-to-Noise density. Zero implies invalid cn0.

uint8[]  lock               # Lock timer.
uint8[]  flags              # Measurement status flags.
uint8[]  sid_sat            # Constellation-specific satellite identifier.
uint8[]  sid_code           # Signal constellation, band and code.