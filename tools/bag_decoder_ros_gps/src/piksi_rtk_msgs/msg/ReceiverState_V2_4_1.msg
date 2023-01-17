# Miscellaneous information form different SBP messages

# Message definition based on libsbp v2.4.1


Header header

uint8 num_sat                   # Number of satellites.
bool rtk_mode_fix 	            # 1 = Fixed, 0 = Float.
uint8[] sat                     # Constellation-specific satellite identifier.
uint8[] cn0                     # Carrier-to-Noise density. Zero implies invalid cn0 [dB Hz / 4].
uint8 system_error              # System Error Flag.
uint8 io_error                  # IO Error Flag.
uint8 swift_nap_error           # SwiftNAP Error Flag.
uint8 external_antenna_present  # External Antenna Present Flag.
uint8 num_gps_sat               # Number of GPS satellites.
uint8[] cn0_gps                 # Carrier-to-Noise density of GPS satellites. Zero implies invalid cn0 [dB Hz / 4].
uint8 num_sbas_sat              # Number of SBAS satellites.
uint8[] cn0_sbas                # Carrier-to-Noise density of SBAS satellites. Zero implies invalid cn0 [dB Hz / 4].
uint8 num_glonass_sat           # Number of GLONASS satellites.
uint8[] cn0_glonass             # Carrier-to-Noise density of GLONASS satellites. Zero implies invalid cn0 [dB Hz / 4].
uint8 num_bds_sat               # Number of BEIDOU satellites.
uint8[] cn0_bds                 # Carrier-to-Noise density of Beidou satellites. Zero implies invalid cn0 [dB Hz / 4].
uint8 num_gal_sat               # Number of GALILEO satellites.
uint8[] cn0_gal                 # Carrier-to-Noise density of GALILEO satellites. Zero implies invalid cn0 [dB Hz / 4].
string fix_mode                 # Invalid, Single Point Position (SPP), Differential GNSS (DGNSS), Float RTK, Fixed RTK.


# codes available in message "MeasurementState.msg"

string STR_FIX_MODE_INVALID        = Invalid
string STR_FIX_MODE_SPP            = SPP
string STR_FIX_MODE_DGNSS          = DGNSS
string STR_FIX_MODE_FLOAT_RTK      = FLOAT_RTK
string STR_FIX_MODE_FIXED_RTK      = FIXED_RTK
string STR_FIX_MODE_DEAD_RECKONING = DEAD_RECKONING
string STR_FIX_MODE_SBAS           = SBAS
string STR_FIX_MODE_UNKNOWN        = Unknown
