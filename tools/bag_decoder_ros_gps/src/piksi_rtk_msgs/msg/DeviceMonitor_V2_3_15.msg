# This message contains temperature and voltage level measurements from the processor's monitoring
# system and the RF frontend die temperature if available.

# Message definition based on libsbp v2.2.15

Header header

int16 dev_vin               # Device V_in [V / 1000].
int16 cpu_vint              # Processor V_int [V / 1000].
int16 cpu_vaux              # Processor V_aux [V / 1000].
int16 cpu_temperature       # Processor temperature [degrees C / 100].
int16 fe_temperature        # Fronted temperature (if available) [degrees C / 100].
