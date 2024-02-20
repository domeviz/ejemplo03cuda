########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(openal-soft_COMPONENT_NAMES "")
set(openal-soft_FIND_DEPENDENCY_NAMES "")

########### VARIABLES #######################################################################
#############################################################################################
set(openal-soft_PACKAGE_FOLDER_DEBUG "C:/Users/fing.labcom/.conan2/p/b/openad5184acb55d28/p")
set(openal-soft_BUILD_MODULES_PATHS_DEBUG "${openal-soft_PACKAGE_FOLDER_DEBUG}/lib/cmake/conan-official-openal-soft-variables.cmake")


set(openal-soft_INCLUDE_DIRS_DEBUG )
set(openal-soft_RES_DIRS_DEBUG )
set(openal-soft_DEFINITIONS_DEBUG "-DAL_LIBTYPE_STATIC")
set(openal-soft_SHARED_LINK_FLAGS_DEBUG )
set(openal-soft_EXE_LINK_FLAGS_DEBUG )
set(openal-soft_OBJECTS_DEBUG )
set(openal-soft_COMPILE_DEFINITIONS_DEBUG "AL_LIBTYPE_STATIC")
set(openal-soft_COMPILE_OPTIONS_C_DEBUG )
set(openal-soft_COMPILE_OPTIONS_CXX_DEBUG )
set(openal-soft_LIB_DIRS_DEBUG "${openal-soft_PACKAGE_FOLDER_DEBUG}/lib")
set(openal-soft_BIN_DIRS_DEBUG )
set(openal-soft_LIBRARY_TYPE_DEBUG STATIC)
set(openal-soft_IS_HOST_WINDOWS_DEBUG 1)
set(openal-soft_LIBS_DEBUG OpenAL32)
set(openal-soft_SYSTEM_LIBS_DEBUG winmm ole32 shell32 user32)
set(openal-soft_FRAMEWORK_DIRS_DEBUG )
set(openal-soft_FRAMEWORKS_DEBUG )
set(openal-soft_BUILD_DIRS_DEBUG )
set(openal-soft_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(openal-soft_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openal-soft_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openal-soft_COMPILE_OPTIONS_C_DEBUG}>")
set(openal-soft_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openal-soft_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openal-soft_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openal-soft_EXE_LINK_FLAGS_DEBUG}>")


set(openal-soft_COMPONENTS_DEBUG )