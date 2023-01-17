# Raw data from the Inertial Measurement Unit, containing accelerometer and gyroscope readings.

Header header

uint32 tow    # Milliseconds since start of GPS week. If the high bit is set, the time is unknown or invalid.
uint8 tow_f   # Milliseconds since start of GPS week, fractional part.
int16 acc_x   # Acceleration in the body frame X axis.
int16 acc_y   # Acceleration in the body frame Y axis.
int16 acc_z   # Acceleration in the body frame Z axis.
int16 gyr_x   # Angular rate around the body frame X axis.
int16 gyr_y   # Angular rate around the body frame Y axis.
int16 gyr_z   # Angular rate around the body frame Z axis.
