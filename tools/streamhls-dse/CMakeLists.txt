configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/streamhls-dse
  ${LLVM_RUNTIME_OUTPUT_INTDIR}/streamhls-dse
  COPYONLY
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/streamhls_dse.py
  ${LLVM_RUNTIME_OUTPUT_INTDIR}/streamhls_dse.py
  COPYONLY
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/dse_problem_v1.schema.json
  ${LLVM_RUNTIME_OUTPUT_INTDIR}/dse_problem_v1.schema.json
  COPYONLY
)
