cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
project(naive-bayes)

# This tells the compiler to not aggressively optimize and
# to include debugging information so that the debugger
# can properly read what's going on.
set(CMAKE_BUILD_TYPE Debug)

# Let's ensure -std=c++xx instead of -std=g++xx
set(CMAKE_CXX_EXTENSIONS OFF)

# Add Boost
#add_subdirectory(boost_1_61_0)
#find_package(Boost 1.61.0 COMPONENTS program_options REQUIRED)
#if(Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIRS})
#    add_executable(main ./apps/train_model_main.cc)
#    target_link_libraries( main program_options)
#endif()


# Let's nicely support folders in IDE's
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Warning flags
if(MSVC)
    # warning level 3 and all warnings as errors
    add_compile_options(/W3 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wpedantic -Werror)
endif()

# FetchContent added in CMake 3.11, downloads during the configure step
include(FetchContent)

# FetchContent_MakeAvailable was not added until CMake 3.14
if(${CMAKE_VERSION} VERSION_LESS 3.14)
    include(cmake/add_FetchContent_MakeAvailable.cmake)
endif()

FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
)

# Adds Catch2 testing library
FetchContent_GetProperties(catch2)
if(NOT catch2_POPULATED)
    FetchContent_Populate(catch2)
    add_library(catch2 INTERFACE)
    target_include_directories(catch2 INTERFACE ${catch2_SOURCE_DIR}/single_include)
endif()

get_filename_component(CINDER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)
get_filename_component(APP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/" ABSOLUTE)

include("${CINDER_PATH}/proj/cmake/modules/cinderMakeApp.cmake")

list(APPEND CORE_SOURCE_FILES src/core/rename_this_file.cc
        src/image.cc
        src/model.cc
        src/model_loader.cc
        src/model_classifier.cc)

list(APPEND SOURCE_FILES    ${CORE_SOURCE_FILES}
        src/visualizer/naive_bayes_app.cc
        src/visualizer/sketchpad.cc)

list(APPEND TEST_FILES tests/test_main.cc
        tests/model_tests.cc
        tests/model_loader_tests.cc
        tests/model_classifier_tests.cc)



#target_link_libraries(
#        train-model
#        ${Boost_PROGRAM_OPTIONS_LIBRARY}
#)
#message(${Boost_PROGRAM_OPTIONS_LIBRARY})

# Add Boost

find_package(Boost 1.73.0 COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    #    include_directories(${Boost_INCLUDE_DIRS})
    #    add_executable(train-model apps/train_model_main.cc)
    add_executable(train-model apps/train_model_main.cc ${CORE_SOURCE_FILES})
    target_include_directories(train-model PRIVATE include)
    target_link_libraries(train-model PUBLIC Boost::program_options)
endif()


ci_make_app(
        APP_NAME        sketchpad-classifier
        CINDER_PATH     ${CINDER_PATH}
        SOURCES         apps/cinder_app_main.cc ${SOURCE_FILES}
        INCLUDES        include
)

ci_make_app(
        APP_NAME        naive-bayes-test
        CINDER_PATH     ${CINDER_PATH}
        SOURCES         tests/test_main.cc ${SOURCE_FILES} ${TEST_FILES}
        INCLUDES        include
        LIBRARIES       catch2
)

if(MSVC)
    set_property(TARGET naive-bayes-test APPEND_STRING PROPERTY LINK_FLAGS " /SUBSYSTEM:CONSOLE")
endif()