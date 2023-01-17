# The tracking message returns a variable-length array of tracking channel states. It reports status and
# carrier-to-noise density measurements for all tracked satellites.

# Message definition based on libsbp v2.2.15

Header header

uint8[] sat     # Constellation-specific satellite identifier.
uint8[] code    # Signal constellation, band and code.
uint8[] fcn     # Frequency channel number (GLONASS only).
uint8[] cn0     # Carrier-to-Noise density. Zero implies invalid cn0 [dB Hz / 4].
