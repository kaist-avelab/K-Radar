# The tracking message returns a variable-length array of tracking channel states. It reports status and
# carrier-to-noise density measurements for all tracked satellites.

# Message definition based on libsbp v2.4.1

Header header

uint8[] sat     # Constellation-specific satellite identifier.
uint8[] code    # Signal constellation, band and code.
uint8[] cn0     # Carrier-to-Noise density. Zero implies invalid cn0 [dB Hz / 4].


uint8 CODE_GPS_L1CA     = 0
uint8 CODE_GPS_L2CM     = 1
uint8 CODE_SBAS_L1CA    = 2
uint8 CODE_GLO_L1CA     = 3
uint8 CODE_GLO_L2CA     = 4
uint8 CODE_GPS_L1P      = 5
uint8 CODE_GPS_L2P      = 6
uint8 CODE_BDS2_B1      = 12
uint8 CODE_BDS2_B2      = 13
uint8 CODE_GAL_E1B      = 14
uint8 CODE_GAL_E7I      = 20
