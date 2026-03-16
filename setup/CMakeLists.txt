add_executable(device_query device_query.cu)
install(TARGETS device_query RUNTIME DESTINATION bin COMPONENT device_query)
