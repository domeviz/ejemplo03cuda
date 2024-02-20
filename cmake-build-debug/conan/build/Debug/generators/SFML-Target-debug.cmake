# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(sfml_FRAMEWORKS_FOUND_DEBUG "") # Will be filled later
conan_find_apple_frameworks(sfml_FRAMEWORKS_FOUND_DEBUG "${sfml_FRAMEWORKS_DEBUG}" "${sfml_FRAMEWORK_DIRS_DEBUG}")

set(sfml_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET sfml_DEPS_TARGET)
    add_library(sfml_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET sfml_DEPS_TARGET
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${sfml_FRAMEWORKS_FOUND_DEBUG}>
             $<$<CONFIG:Debug>:${sfml_SYSTEM_LIBS_DEBUG}>
             $<$<CONFIG:Debug>:sfml-system;opengl::opengl;sfml-window;Freetype::Freetype;flac::flac;OpenAL::OpenAL;vorbis::vorbis>
             APPEND)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### sfml_DEPS_TARGET to all of them
conan_package_library_targets("${sfml_LIBS_DEBUG}"    # libraries
                              "${sfml_LIB_DIRS_DEBUG}" # package_libdir
                              "${sfml_BIN_DIRS_DEBUG}" # package_bindir
                              "${sfml_LIBRARY_TYPE_DEBUG}"
                              "${sfml_IS_HOST_WINDOWS_DEBUG}"
                              sfml_DEPS_TARGET
                              sfml_LIBRARIES_TARGETS  # out_libraries_targets
                              "_DEBUG"
                              "sfml"    # package_name
                              "${sfml_NO_SONAME_MODE_DEBUG}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${sfml_BUILD_DIRS_DEBUG} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Debug ########################################

    ########## COMPONENT sfml-audio #############

        set(sfml_sfml-audio_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(sfml_sfml-audio_FRAMEWORKS_FOUND_DEBUG "${sfml_sfml-audio_FRAMEWORKS_DEBUG}" "${sfml_sfml-audio_FRAMEWORK_DIRS_DEBUG}")

        set(sfml_sfml-audio_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET sfml_sfml-audio_DEPS_TARGET)
            add_library(sfml_sfml-audio_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET sfml_sfml-audio_DEPS_TARGET
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_DEPENDENCIES_DEBUG}>
                     APPEND)

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'sfml_sfml-audio_DEPS_TARGET' to all of them
        conan_package_library_targets("${sfml_sfml-audio_LIBS_DEBUG}"
                              "${sfml_sfml-audio_LIB_DIRS_DEBUG}"
                              "${sfml_sfml-audio_BIN_DIRS_DEBUG}" # package_bindir
                              "${sfml_sfml-audio_LIBRARY_TYPE_DEBUG}"
                              "${sfml_sfml-audio_IS_HOST_WINDOWS_DEBUG}"
                              sfml_sfml-audio_DEPS_TARGET
                              sfml_sfml-audio_LIBRARIES_TARGETS
                              "_DEBUG"
                              "sfml_sfml-audio"
                              "${sfml_sfml-audio_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET sfml-audio
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_LIBRARIES_TARGETS}>
                     APPEND)

        if("${sfml_sfml-audio_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET sfml-audio
                         PROPERTY INTERFACE_LINK_LIBRARIES
                         sfml_sfml-audio_DEPS_TARGET
                         APPEND)
        endif()

        set_property(TARGET sfml-audio PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_LINKER_FLAGS_DEBUG}> APPEND)
        set_property(TARGET sfml-audio PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_INCLUDE_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-audio PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_LIB_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-audio PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_COMPILE_DEFINITIONS_DEBUG}> APPEND)
        set_property(TARGET sfml-audio PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-audio_COMPILE_OPTIONS_DEBUG}> APPEND)

    ########## COMPONENT sfml-network #############

        set(sfml_sfml-network_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(sfml_sfml-network_FRAMEWORKS_FOUND_DEBUG "${sfml_sfml-network_FRAMEWORKS_DEBUG}" "${sfml_sfml-network_FRAMEWORK_DIRS_DEBUG}")

        set(sfml_sfml-network_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET sfml_sfml-network_DEPS_TARGET)
            add_library(sfml_sfml-network_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET sfml_sfml-network_DEPS_TARGET
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-network_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-network_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-network_DEPENDENCIES_DEBUG}>
                     APPEND)

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'sfml_sfml-network_DEPS_TARGET' to all of them
        conan_package_library_targets("${sfml_sfml-network_LIBS_DEBUG}"
                              "${sfml_sfml-network_LIB_DIRS_DEBUG}"
                              "${sfml_sfml-network_BIN_DIRS_DEBUG}" # package_bindir
                              "${sfml_sfml-network_LIBRARY_TYPE_DEBUG}"
                              "${sfml_sfml-network_IS_HOST_WINDOWS_DEBUG}"
                              sfml_sfml-network_DEPS_TARGET
                              sfml_sfml-network_LIBRARIES_TARGETS
                              "_DEBUG"
                              "sfml_sfml-network"
                              "${sfml_sfml-network_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET sfml-network
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-network_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-network_LIBRARIES_TARGETS}>
                     APPEND)

        if("${sfml_sfml-network_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET sfml-network
                         PROPERTY INTERFACE_LINK_LIBRARIES
                         sfml_sfml-network_DEPS_TARGET
                         APPEND)
        endif()

        set_property(TARGET sfml-network PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-network_LINKER_FLAGS_DEBUG}> APPEND)
        set_property(TARGET sfml-network PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-network_INCLUDE_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-network PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-network_LIB_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-network PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-network_COMPILE_DEFINITIONS_DEBUG}> APPEND)
        set_property(TARGET sfml-network PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-network_COMPILE_OPTIONS_DEBUG}> APPEND)

    ########## COMPONENT sfml-graphics #############

        set(sfml_sfml-graphics_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(sfml_sfml-graphics_FRAMEWORKS_FOUND_DEBUG "${sfml_sfml-graphics_FRAMEWORKS_DEBUG}" "${sfml_sfml-graphics_FRAMEWORK_DIRS_DEBUG}")

        set(sfml_sfml-graphics_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET sfml_sfml-graphics_DEPS_TARGET)
            add_library(sfml_sfml-graphics_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET sfml_sfml-graphics_DEPS_TARGET
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_DEPENDENCIES_DEBUG}>
                     APPEND)

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'sfml_sfml-graphics_DEPS_TARGET' to all of them
        conan_package_library_targets("${sfml_sfml-graphics_LIBS_DEBUG}"
                              "${sfml_sfml-graphics_LIB_DIRS_DEBUG}"
                              "${sfml_sfml-graphics_BIN_DIRS_DEBUG}" # package_bindir
                              "${sfml_sfml-graphics_LIBRARY_TYPE_DEBUG}"
                              "${sfml_sfml-graphics_IS_HOST_WINDOWS_DEBUG}"
                              sfml_sfml-graphics_DEPS_TARGET
                              sfml_sfml-graphics_LIBRARIES_TARGETS
                              "_DEBUG"
                              "sfml_sfml-graphics"
                              "${sfml_sfml-graphics_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET sfml-graphics
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_LIBRARIES_TARGETS}>
                     APPEND)

        if("${sfml_sfml-graphics_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET sfml-graphics
                         PROPERTY INTERFACE_LINK_LIBRARIES
                         sfml_sfml-graphics_DEPS_TARGET
                         APPEND)
        endif()

        set_property(TARGET sfml-graphics PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_LINKER_FLAGS_DEBUG}> APPEND)
        set_property(TARGET sfml-graphics PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_INCLUDE_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-graphics PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_LIB_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-graphics PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_COMPILE_DEFINITIONS_DEBUG}> APPEND)
        set_property(TARGET sfml-graphics PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-graphics_COMPILE_OPTIONS_DEBUG}> APPEND)

    ########## COMPONENT sfml-window #############

        set(sfml_sfml-window_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(sfml_sfml-window_FRAMEWORKS_FOUND_DEBUG "${sfml_sfml-window_FRAMEWORKS_DEBUG}" "${sfml_sfml-window_FRAMEWORK_DIRS_DEBUG}")

        set(sfml_sfml-window_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET sfml_sfml-window_DEPS_TARGET)
            add_library(sfml_sfml-window_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET sfml_sfml-window_DEPS_TARGET
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-window_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-window_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-window_DEPENDENCIES_DEBUG}>
                     APPEND)

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'sfml_sfml-window_DEPS_TARGET' to all of them
        conan_package_library_targets("${sfml_sfml-window_LIBS_DEBUG}"
                              "${sfml_sfml-window_LIB_DIRS_DEBUG}"
                              "${sfml_sfml-window_BIN_DIRS_DEBUG}" # package_bindir
                              "${sfml_sfml-window_LIBRARY_TYPE_DEBUG}"
                              "${sfml_sfml-window_IS_HOST_WINDOWS_DEBUG}"
                              sfml_sfml-window_DEPS_TARGET
                              sfml_sfml-window_LIBRARIES_TARGETS
                              "_DEBUG"
                              "sfml_sfml-window"
                              "${sfml_sfml-window_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET sfml-window
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-window_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-window_LIBRARIES_TARGETS}>
                     APPEND)

        if("${sfml_sfml-window_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET sfml-window
                         PROPERTY INTERFACE_LINK_LIBRARIES
                         sfml_sfml-window_DEPS_TARGET
                         APPEND)
        endif()

        set_property(TARGET sfml-window PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-window_LINKER_FLAGS_DEBUG}> APPEND)
        set_property(TARGET sfml-window PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-window_INCLUDE_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-window PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-window_LIB_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-window PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-window_COMPILE_DEFINITIONS_DEBUG}> APPEND)
        set_property(TARGET sfml-window PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-window_COMPILE_OPTIONS_DEBUG}> APPEND)

    ########## COMPONENT sfml-main #############

        set(sfml_sfml-main_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(sfml_sfml-main_FRAMEWORKS_FOUND_DEBUG "${sfml_sfml-main_FRAMEWORKS_DEBUG}" "${sfml_sfml-main_FRAMEWORK_DIRS_DEBUG}")

        set(sfml_sfml-main_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET sfml_sfml-main_DEPS_TARGET)
            add_library(sfml_sfml-main_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET sfml_sfml-main_DEPS_TARGET
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-main_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-main_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-main_DEPENDENCIES_DEBUG}>
                     APPEND)

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'sfml_sfml-main_DEPS_TARGET' to all of them
        conan_package_library_targets("${sfml_sfml-main_LIBS_DEBUG}"
                              "${sfml_sfml-main_LIB_DIRS_DEBUG}"
                              "${sfml_sfml-main_BIN_DIRS_DEBUG}" # package_bindir
                              "${sfml_sfml-main_LIBRARY_TYPE_DEBUG}"
                              "${sfml_sfml-main_IS_HOST_WINDOWS_DEBUG}"
                              sfml_sfml-main_DEPS_TARGET
                              sfml_sfml-main_LIBRARIES_TARGETS
                              "_DEBUG"
                              "sfml_sfml-main"
                              "${sfml_sfml-main_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET sfml-main
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-main_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-main_LIBRARIES_TARGETS}>
                     APPEND)

        if("${sfml_sfml-main_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET sfml-main
                         PROPERTY INTERFACE_LINK_LIBRARIES
                         sfml_sfml-main_DEPS_TARGET
                         APPEND)
        endif()

        set_property(TARGET sfml-main PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-main_LINKER_FLAGS_DEBUG}> APPEND)
        set_property(TARGET sfml-main PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-main_INCLUDE_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-main PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-main_LIB_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-main PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-main_COMPILE_DEFINITIONS_DEBUG}> APPEND)
        set_property(TARGET sfml-main PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-main_COMPILE_OPTIONS_DEBUG}> APPEND)

    ########## COMPONENT sfml-system #############

        set(sfml_sfml-system_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(sfml_sfml-system_FRAMEWORKS_FOUND_DEBUG "${sfml_sfml-system_FRAMEWORKS_DEBUG}" "${sfml_sfml-system_FRAMEWORK_DIRS_DEBUG}")

        set(sfml_sfml-system_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET sfml_sfml-system_DEPS_TARGET)
            add_library(sfml_sfml-system_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET sfml_sfml-system_DEPS_TARGET
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-system_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-system_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-system_DEPENDENCIES_DEBUG}>
                     APPEND)

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'sfml_sfml-system_DEPS_TARGET' to all of them
        conan_package_library_targets("${sfml_sfml-system_LIBS_DEBUG}"
                              "${sfml_sfml-system_LIB_DIRS_DEBUG}"
                              "${sfml_sfml-system_BIN_DIRS_DEBUG}" # package_bindir
                              "${sfml_sfml-system_LIBRARY_TYPE_DEBUG}"
                              "${sfml_sfml-system_IS_HOST_WINDOWS_DEBUG}"
                              sfml_sfml-system_DEPS_TARGET
                              sfml_sfml-system_LIBRARIES_TARGETS
                              "_DEBUG"
                              "sfml_sfml-system"
                              "${sfml_sfml-system_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET sfml-system
                     PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${sfml_sfml-system_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${sfml_sfml-system_LIBRARIES_TARGETS}>
                     APPEND)

        if("${sfml_sfml-system_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET sfml-system
                         PROPERTY INTERFACE_LINK_LIBRARIES
                         sfml_sfml-system_DEPS_TARGET
                         APPEND)
        endif()

        set_property(TARGET sfml-system PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-system_LINKER_FLAGS_DEBUG}> APPEND)
        set_property(TARGET sfml-system PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-system_INCLUDE_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-system PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${sfml_sfml-system_LIB_DIRS_DEBUG}> APPEND)
        set_property(TARGET sfml-system PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-system_COMPILE_DEFINITIONS_DEBUG}> APPEND)
        set_property(TARGET sfml-system PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${sfml_sfml-system_COMPILE_OPTIONS_DEBUG}> APPEND)

    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET sfml::sfml PROPERTY INTERFACE_LINK_LIBRARIES sfml-audio APPEND)
    set_property(TARGET sfml::sfml PROPERTY INTERFACE_LINK_LIBRARIES sfml-network APPEND)
    set_property(TARGET sfml::sfml PROPERTY INTERFACE_LINK_LIBRARIES sfml-graphics APPEND)
    set_property(TARGET sfml::sfml PROPERTY INTERFACE_LINK_LIBRARIES sfml-window APPEND)
    set_property(TARGET sfml::sfml PROPERTY INTERFACE_LINK_LIBRARIES sfml-main APPEND)
    set_property(TARGET sfml::sfml PROPERTY INTERFACE_LINK_LIBRARIES sfml-system APPEND)

########## For the modules (FindXXX)
set(sfml_LIBRARIES_DEBUG sfml::sfml)
