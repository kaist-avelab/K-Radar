# This dilution of precision (DOP) message describes the effect of navigation satellite geometry on positional
# measurement precision.

Header header

uint32 tow  # GPS Time of Week [ms].
uint16 gdop # Geometric Dilution of Precision.
uint16 pdop # Position Dilution of Precision.
uint16 tdop # Time Dilution of Precision.
uint16 hdop # Horizontal Dilution of Precision.
uint16 vdop # Vertical Dilution of Precision.
uint8 flags # Indicates the position solution with which the DOPS message corresponds.
