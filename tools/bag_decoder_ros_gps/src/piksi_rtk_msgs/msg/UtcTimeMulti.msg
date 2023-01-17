# This message reports the Universal Coordinated Time (UTC). Note the flags which
# indicate the source of the UTC offset value and source of the time fix.

Header header

uint8 flags   # Indicates source and time validity
uint32 tow    # GPS time of week rounded to the nearest millisecond
uint16 year   # Year
uint8 month   # Month (range 1 .. 12)
uint8 day     # days in the month (range 1-31)
uint8 hours   # hours of day (range 0-23)
uint8 minutes # minutes of hour (range 0-59)
uint8 seconds # seconds of minute (range 0-60) rounded down
uint32 ns     # nanoseconds of second (range 0-999999999)
