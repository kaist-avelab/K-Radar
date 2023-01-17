# The UART message reports data latency and throughput of the UART channels providing SBP I/O.

# Message definition based on libsbp v2.3.15, only some field were selected for the original SBP message (same ones
# used in piksi_tools/console/system_monitor_view.py, function "uart_state_callback").

Header header

int32 latency_avg               # Average latency [ms].
int32 latency_lmin              # Minimum latency [ms].
int32 latency_lmax              # Maximum latency [ms].
int32 latency_current           # Smoothed estimate of the current latency [ms].
int32 obs_period_avg            # Average period [ms].
int32 obs_period_pmin           # Minimum period [ms].
int32 obs_period_pmax           # Maximum period [ms].
int32 obs_period_current        # Smoothed estimate of the current preiod [ms].
