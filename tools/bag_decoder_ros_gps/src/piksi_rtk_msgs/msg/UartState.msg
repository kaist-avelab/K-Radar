# The UART message reports data latency and throughput of the UART channels providing SBP I/O.
# On the default Piksi configuration, UARTs A and B are used for telemetry radios, but can also be host
# access ports for embedded hosts, or other interfaces in future. The reported percentage values must
# be normalized. Observations latency and period can be used to assess the health of the differential
# corrections link. Latency provides the timeliness of received base observations while the period indicates
# their likelihood of transmission.

Header header

float64 uart_a_tx_throughput      # UART transmit throughput [kB/s].
float64 uart_a_rx_throughput      # UART transmit throughput [kB/s].
uint16 uart_a_crc_error_count     # UART CRC error count.
uint16 uart_a_io_error_count      # UART IO error count.
uint8 uart_a_tx_buffer_level      # UART transmit buffer percentage utilization (ranges from 0 to 255).
uint8 uart_a_rx_buffer_level      # UART transmit buffer percentage utilization (ranges from 0 to 255).

float64 uart_b_tx_throughput      # UART transmit throughput [kB/s].
float64 uart_b_rx_throughput      # UART transmit throughput [kB/s].
uint16 uart_b_crc_error_count     # UART CRC error count.
uint16 uart_b_io_error_count      # UART IO error count.
uint8 uart_b_tx_buffer_level      # UART transmit buffer percentage utilization (ranges from 0 to 255).
uint8 uart_b_rx_buffer_level      # UART transmit buffer percentage utilization (ranges from 0 to 255).

float64 uart_ftdi_tx_throughput   # UART transmit throughput [kB/s].
float64 uart_ftdi_rx_throughput   # UART receive throughput [kB/s].
uint16 uart_ftdi_crc_error_count  # UART CRC error count.
uint16 uart_ftdi_io_error_count   # UART IO error count.
uint8 uart_ftdi_tx_buffer_level   # UART transmit buffer percentage utilization (ranges from 0 to 255).
uint8 uart_ftdi_rx_buffer_level   # UART receive buffer percentage utilization (ranges from 0 to 255).

int32 latency_avg                 # Average latency [ms].
int32 latency_lmin                # Minimum latency [ms].
int32 latency_lmax                # Maximum latency [ms].
int32 latency_current             # Smoothed estimate of the current latency [ms].

int32 obs_period_avg              # Average period [ms].
int32 obs_period_pmin             # Minimum period [ms].
int32 obs_period_pmax             # Maximum period [ms].
int32 obs_period_current          # Smoothed estimate of the current period [ms].
