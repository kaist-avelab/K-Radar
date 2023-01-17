# Reports detection of an external event, the GPS time it occurred, which pin it
# was and whether it was rising or falling.

# Modified to contain ROS time.

std_msgs/Time stamp
std_msgs/Bool pin_value
std_msgs/Bool quality
std_msgs/UInt8 pin

uint8 PIN_VALUE_LOW   = 0   # Falling edge.
uint8 PIN_VALUE_HIGH  = 1   # Rising edge
uint8 QUALITY_UNKNOWN = 0   # No NAV solution.
uint8 QUALITY_GOOD    = 1   # 1 Microsecond accuracy.
