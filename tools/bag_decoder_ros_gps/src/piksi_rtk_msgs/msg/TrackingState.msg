# The tracking message returns a variable-length array of tracking channel states. It reports status and
# carrier-to-noise density measurements for all tracked satellites.

Header header

uint8[] state   # Status of tracking channel.
uint16[] sat    # Constellation-specific satellite identifier.
uint8[] code    # Signal constellation, band and code.
float32[] cn0   # Carrier-to-noise density.
