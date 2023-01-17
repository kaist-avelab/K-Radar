# Miscellaneous information form different SBP messages

Header header

uint8 num_sat                   # Number of satellites.
bool rtk_mode_fix 	            # 1 = Fixed, 0 = Float.
uint16[] sat                    # Constellation-specific satellite identifier.
float32[] cn0                   # Carrier-to-noise density.
uint8[] tracking_running        # Status of tracking channel.
uint8 system_error              # System Error Flag.
uint8 io_error                  # IO Error Flag.
uint8 swift_nap_error           # SwiftNAP Error Flag.
uint8 external_antenna_present  # External Antenna Present Flag.
