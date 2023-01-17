# Auxiliary data specific to a particular IMU. The 'imu type' field will always be consistent but the rest of
# the payload is device specific and depends on the value of 'imu type'.

Header header

uint8 imu_type  # IMU type.
int16 temp      # Raw IMU temperature.
uint8 imu_conf  # IMU configuration.
