########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND sfml_COMPONENT_NAMES sfml-system sfml-main sfml-window sfml-graphics sfml-network sfml-audio)
list(REMOVE_DUPLICATES sfml_COMPONENT_NAMES)
list(APPEND sfml_FIND_DEPENDENCY_NAMES opengl_system freetype flac OpenAL Vorbis)
list(REMOVE_DUPLICATES sfml_FIND_DEPENDENCY_NAMES)
set(opengl_system_FIND_MODE "NO_MODULE")
set(freetype_FIND_MODE "NO_MODULE")
set(flac_FIND_MODE "NO_MODULE")
set(OpenAL_FIND_MODE "NO_MODULE")
set(Vorbis_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(sfml_PACKAGE_FOLDER_DEBUG "C:/Users/fing.labcom/.conan2/p/b/sfmlead451e561c43/p")
set(sfml_BUILD_MODULES_PATHS_DEBUG )


set(sfml_INCLUDE_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/include")
set(sfml_RES_DIRS_DEBUG )
set(sfml_DEFINITIONS_DEBUG "-DSFML_STATIC")
set(sfml_SHARED_LINK_FLAGS_DEBUG )
set(sfml_EXE_LINK_FLAGS_DEBUG )
set(sfml_OBJECTS_DEBUG )
set(sfml_COMPILE_DEFINITIONS_DEBUG "SFML_STATIC")
set(sfml_COMPILE_OPTIONS_C_DEBUG )
set(sfml_COMPILE_OPTIONS_CXX_DEBUG )
set(sfml_LIB_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/lib")
set(sfml_BIN_DIRS_DEBUG )
set(sfml_LIBRARY_TYPE_DEBUG STATIC)
set(sfml_IS_HOST_WINDOWS_DEBUG 1)
set(sfml_LIBS_DEBUG sfml-audio-s-d sfml-network-s-d sfml-graphics-s-d sfml-window-s-d sfml-main-d sfml-system-s-d)
set(sfml_SYSTEM_LIBS_DEBUG ws2_32 gdi32 winmm)
set(sfml_FRAMEWORK_DIRS_DEBUG )
set(sfml_FRAMEWORKS_DEBUG )
set(sfml_BUILD_DIRS_DEBUG )
set(sfml_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(sfml_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_COMPILE_OPTIONS_C_DEBUG}>")
set(sfml_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_EXE_LINK_FLAGS_DEBUG}>")


set(sfml_COMPONENTS_DEBUG sfml-system sfml-main sfml-window sfml-graphics sfml-network sfml-audio)
########### COMPONENT sfml-audio VARIABLES ############################################

set(sfml_sfml-audio_INCLUDE_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/include")
set(sfml_sfml-audio_LIB_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/lib")
set(sfml_sfml-audio_BIN_DIRS_DEBUG )
set(sfml_sfml-audio_LIBRARY_TYPE_DEBUG STATIC)
set(sfml_sfml-audio_IS_HOST_WINDOWS_DEBUG 1)
set(sfml_sfml-audio_RES_DIRS_DEBUG )
set(sfml_sfml-audio_DEFINITIONS_DEBUG "-DSFML_STATIC")
set(sfml_sfml-audio_OBJECTS_DEBUG )
set(sfml_sfml-audio_COMPILE_DEFINITIONS_DEBUG "SFML_STATIC")
set(sfml_sfml-audio_COMPILE_OPTIONS_C_DEBUG "")
set(sfml_sfml-audio_COMPILE_OPTIONS_CXX_DEBUG "")
set(sfml_sfml-audio_LIBS_DEBUG sfml-audio-s-d)
set(sfml_sfml-audio_SYSTEM_LIBS_DEBUG )
set(sfml_sfml-audio_FRAMEWORK_DIRS_DEBUG )
set(sfml_sfml-audio_FRAMEWORKS_DEBUG )
set(sfml_sfml-audio_DEPENDENCIES_DEBUG sfml-system flac::flac OpenAL::OpenAL vorbis::vorbis)
set(sfml_sfml-audio_SHARED_LINK_FLAGS_DEBUG )
set(sfml_sfml-audio_EXE_LINK_FLAGS_DEBUG )
set(sfml_sfml-audio_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-audio_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-audio_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-audio_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-audio_EXE_LINK_FLAGS_DEBUG}>
)
set(sfml_sfml-audio_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-audio_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-audio_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT sfml-network VARIABLES ############################################

set(sfml_sfml-network_INCLUDE_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/include")
set(sfml_sfml-network_LIB_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/lib")
set(sfml_sfml-network_BIN_DIRS_DEBUG )
set(sfml_sfml-network_LIBRARY_TYPE_DEBUG STATIC)
set(sfml_sfml-network_IS_HOST_WINDOWS_DEBUG 1)
set(sfml_sfml-network_RES_DIRS_DEBUG )
set(sfml_sfml-network_DEFINITIONS_DEBUG "-DSFML_STATIC")
set(sfml_sfml-network_OBJECTS_DEBUG )
set(sfml_sfml-network_COMPILE_DEFINITIONS_DEBUG "SFML_STATIC")
set(sfml_sfml-network_COMPILE_OPTIONS_C_DEBUG "")
set(sfml_sfml-network_COMPILE_OPTIONS_CXX_DEBUG "")
set(sfml_sfml-network_LIBS_DEBUG sfml-network-s-d)
set(sfml_sfml-network_SYSTEM_LIBS_DEBUG ws2_32)
set(sfml_sfml-network_FRAMEWORK_DIRS_DEBUG )
set(sfml_sfml-network_FRAMEWORKS_DEBUG )
set(sfml_sfml-network_DEPENDENCIES_DEBUG sfml-system)
set(sfml_sfml-network_SHARED_LINK_FLAGS_DEBUG )
set(sfml_sfml-network_EXE_LINK_FLAGS_DEBUG )
set(sfml_sfml-network_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-network_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-network_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-network_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-network_EXE_LINK_FLAGS_DEBUG}>
)
set(sfml_sfml-network_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-network_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-network_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT sfml-graphics VARIABLES ############################################

set(sfml_sfml-graphics_INCLUDE_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/include")
set(sfml_sfml-graphics_LIB_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/lib")
set(sfml_sfml-graphics_BIN_DIRS_DEBUG )
set(sfml_sfml-graphics_LIBRARY_TYPE_DEBUG STATIC)
set(sfml_sfml-graphics_IS_HOST_WINDOWS_DEBUG 1)
set(sfml_sfml-graphics_RES_DIRS_DEBUG )
set(sfml_sfml-graphics_DEFINITIONS_DEBUG "-DSFML_STATIC")
set(sfml_sfml-graphics_OBJECTS_DEBUG )
set(sfml_sfml-graphics_COMPILE_DEFINITIONS_DEBUG "SFML_STATIC")
set(sfml_sfml-graphics_COMPILE_OPTIONS_C_DEBUG "")
set(sfml_sfml-graphics_COMPILE_OPTIONS_CXX_DEBUG "")
set(sfml_sfml-graphics_LIBS_DEBUG sfml-graphics-s-d)
set(sfml_sfml-graphics_SYSTEM_LIBS_DEBUG )
set(sfml_sfml-graphics_FRAMEWORK_DIRS_DEBUG )
set(sfml_sfml-graphics_FRAMEWORKS_DEBUG )
set(sfml_sfml-graphics_DEPENDENCIES_DEBUG sfml-window Freetype::Freetype)
set(sfml_sfml-graphics_SHARED_LINK_FLAGS_DEBUG )
set(sfml_sfml-graphics_EXE_LINK_FLAGS_DEBUG )
set(sfml_sfml-graphics_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-graphics_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-graphics_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-graphics_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-graphics_EXE_LINK_FLAGS_DEBUG}>
)
set(sfml_sfml-graphics_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-graphics_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-graphics_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT sfml-window VARIABLES ############################################

set(sfml_sfml-window_INCLUDE_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/include")
set(sfml_sfml-window_LIB_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/lib")
set(sfml_sfml-window_BIN_DIRS_DEBUG )
set(sfml_sfml-window_LIBRARY_TYPE_DEBUG STATIC)
set(sfml_sfml-window_IS_HOST_WINDOWS_DEBUG 1)
set(sfml_sfml-window_RES_DIRS_DEBUG )
set(sfml_sfml-window_DEFINITIONS_DEBUG "-DSFML_STATIC")
set(sfml_sfml-window_OBJECTS_DEBUG )
set(sfml_sfml-window_COMPILE_DEFINITIONS_DEBUG "SFML_STATIC")
set(sfml_sfml-window_COMPILE_OPTIONS_C_DEBUG "")
set(sfml_sfml-window_COMPILE_OPTIONS_CXX_DEBUG "")
set(sfml_sfml-window_LIBS_DEBUG sfml-window-s-d)
set(sfml_sfml-window_SYSTEM_LIBS_DEBUG gdi32 winmm)
set(sfml_sfml-window_FRAMEWORK_DIRS_DEBUG )
set(sfml_sfml-window_FRAMEWORKS_DEBUG )
set(sfml_sfml-window_DEPENDENCIES_DEBUG sfml-system opengl::opengl)
set(sfml_sfml-window_SHARED_LINK_FLAGS_DEBUG )
set(sfml_sfml-window_EXE_LINK_FLAGS_DEBUG )
set(sfml_sfml-window_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-window_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-window_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-window_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-window_EXE_LINK_FLAGS_DEBUG}>
)
set(sfml_sfml-window_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-window_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-window_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT sfml-main VARIABLES ############################################

set(sfml_sfml-main_INCLUDE_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/include")
set(sfml_sfml-main_LIB_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/lib")
set(sfml_sfml-main_BIN_DIRS_DEBUG )
set(sfml_sfml-main_LIBRARY_TYPE_DEBUG STATIC)
set(sfml_sfml-main_IS_HOST_WINDOWS_DEBUG 1)
set(sfml_sfml-main_RES_DIRS_DEBUG )
set(sfml_sfml-main_DEFINITIONS_DEBUG "-DSFML_STATIC")
set(sfml_sfml-main_OBJECTS_DEBUG )
set(sfml_sfml-main_COMPILE_DEFINITIONS_DEBUG "SFML_STATIC")
set(sfml_sfml-main_COMPILE_OPTIONS_C_DEBUG "")
set(sfml_sfml-main_COMPILE_OPTIONS_CXX_DEBUG "")
set(sfml_sfml-main_LIBS_DEBUG sfml-main-d)
set(sfml_sfml-main_SYSTEM_LIBS_DEBUG )
set(sfml_sfml-main_FRAMEWORK_DIRS_DEBUG )
set(sfml_sfml-main_FRAMEWORKS_DEBUG )
set(sfml_sfml-main_DEPENDENCIES_DEBUG )
set(sfml_sfml-main_SHARED_LINK_FLAGS_DEBUG )
set(sfml_sfml-main_EXE_LINK_FLAGS_DEBUG )
set(sfml_sfml-main_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-main_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-main_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-main_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-main_EXE_LINK_FLAGS_DEBUG}>
)
set(sfml_sfml-main_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-main_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-main_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT sfml-system VARIABLES ############################################

set(sfml_sfml-system_INCLUDE_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/include")
set(sfml_sfml-system_LIB_DIRS_DEBUG "${sfml_PACKAGE_FOLDER_DEBUG}/lib")
set(sfml_sfml-system_BIN_DIRS_DEBUG )
set(sfml_sfml-system_LIBRARY_TYPE_DEBUG STATIC)
set(sfml_sfml-system_IS_HOST_WINDOWS_DEBUG 1)
set(sfml_sfml-system_RES_DIRS_DEBUG )
set(sfml_sfml-system_DEFINITIONS_DEBUG "-DSFML_STATIC")
set(sfml_sfml-system_OBJECTS_DEBUG )
set(sfml_sfml-system_COMPILE_DEFINITIONS_DEBUG "SFML_STATIC")
set(sfml_sfml-system_COMPILE_OPTIONS_C_DEBUG "")
set(sfml_sfml-system_COMPILE_OPTIONS_CXX_DEBUG "")
set(sfml_sfml-system_LIBS_DEBUG sfml-system-s-d)
set(sfml_sfml-system_SYSTEM_LIBS_DEBUG winmm)
set(sfml_sfml-system_FRAMEWORK_DIRS_DEBUG )
set(sfml_sfml-system_FRAMEWORKS_DEBUG )
set(sfml_sfml-system_DEPENDENCIES_DEBUG )
set(sfml_sfml-system_SHARED_LINK_FLAGS_DEBUG )
set(sfml_sfml-system_EXE_LINK_FLAGS_DEBUG )
set(sfml_sfml-system_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-system_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-system_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-system_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-system_EXE_LINK_FLAGS_DEBUG}>
)
set(sfml_sfml-system_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-system_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-system_COMPILE_OPTIONS_C_DEBUG}>")