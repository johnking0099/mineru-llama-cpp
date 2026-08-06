"""Log verbosity thresholds for Engine(verbosity=...).

Values and names are copied verbatim from llama.cpp's common/log.h
LOG_LEVEL_* macros -- passed straight through to
common_log_set_verbosity_thold() in the C++ layer (see engine_core.cpp),
so keeping the same names/values here avoids a second source of truth.

A message is printed when its own level is <= the configured threshold, so
higher constants mean progressively more output: OUTPUT/ERROR/WARN always
show (they're what CLI tools print by default); INFO is llama.cpp's own
default (matches common_params::verbosity's default of 3); TRACE and DEBUG
add llama.cpp's internal tracing/debug lines.
"""

LOG_LEVEL_OUTPUT = 0
LOG_LEVEL_ERROR = 1
LOG_LEVEL_WARN = 2
LOG_LEVEL_INFO = 3
LOG_LEVEL_TRACE = 4
LOG_LEVEL_DEBUG = 5
