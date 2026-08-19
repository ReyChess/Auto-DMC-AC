cmake_minimum_required(VERSION 3.16)
project(AutoDMCAC LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(configure_c_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /O2)
  else()
    target_compile_options(${target} PRIVATE -O3 -msse4.2)
    target_link_libraries(${target} PRIVATE m)
  endif()
endfunction()

# Same exact mining cores are compiled in positive-only and positive-negative modes.
# min_netconf=0.0 is valid and means strict positive evidence Netconf(A=>c)>0;
# negative rules use Netconf(A=>c)<0 only, so Netconf=0 is excluded as independence.
add_executable(miner_global_p src/miner_global.c)
target_compile_definitions(miner_global_p PRIVATE M4_DIRECT_MAXIMAL_POS_NEG=0 M4_GENERATE_NEGATIVES=0)
configure_c_target(miner_global_p)

add_executable(miner_global_pn src/miner_global.c)
target_compile_definitions(miner_global_pn PRIVATE M4_DIRECT_MAXIMAL_POS_NEG=1 M4_GENERATE_NEGATIVES=1)
configure_c_target(miner_global_pn)

add_executable(miner_cns_p src/miner_cns.c)
target_compile_definitions(miner_cns_p PRIVATE M4_DIRECT_MAXIMAL_POS_NEG=0 M4_GENERATE_NEGATIVES=0)
configure_c_target(miner_cns_p)

add_executable(miner_cns_pn src/miner_cns.c)
target_compile_definitions(miner_cns_pn PRIVATE M4_DIRECT_MAXIMAL_POS_NEG=1 M4_GENERATE_NEGATIVES=1)
configure_c_target(miner_cns_pn)

add_executable(dmc_miner_unified src/dmc_miner_dispatcher.cpp)
add_executable(auto_dmc_ac src/auto_dmc_ac.cpp)
if(MSVC)
  target_compile_options(dmc_miner_unified PRIVATE /O2)
  target_compile_options(auto_dmc_ac PRIVATE /O2)
else()
  target_compile_options(dmc_miner_unified PRIVATE -O3)
  target_compile_options(auto_dmc_ac PRIVATE -O3)
endif()
