# The heartbeat message is sent periodically to inform the host or other attached devices that the system
# is running. It is used to monitor system malfunctions. It also contains status flags that indicate to the
# host the status of the system and whether it is operating correctly. Currently, the expected heartbeat
# interval is 1 sec.
# The system error flag is used to indicate that an error has occurred in the system. To determine the
# source of the error, the remaining error flags should be inspected.

Header header

uint8 system_error              # System Error Flag.
uint8 io_error                  # IO Error Flag.
uint8 swift_nap_error           # SwiftNAP Error Flag.
uint8 sbp_minor_version         # SBP Minor Protocol Version Number.
uint8 sbp_major_version         # SBP Major Protocol Version Number.
uint8 external_antenna_present  # External Antenna Present Flag.
