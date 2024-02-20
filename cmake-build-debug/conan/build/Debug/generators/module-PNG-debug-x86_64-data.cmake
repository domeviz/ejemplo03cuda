########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(libpng_COMPONENT_NAMES "")
list(APPEND libpng_FIND_DEPENDENCY_NAMES ZLIB)
list(REMOVE_DUPLICATES libpng_FIND_DEPENDENCY_NAMES)
set(ZLIB_FIND_MODE "MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(libpng_PACKAGE_FOLDER_DEBUG "C:/Users/fing.labcom/.conan2/p/b/libpndf00958a3b51b/p")
set(libpng_BUILD_MODULES_PATHS_DEBUG )


set(libpng_INCLUDE_DIRS_DEBUG )
set(libpng_RES_DIRS_DEBUG )
set(libpng_DEFINITIONS_DEBUG )
set(libpng_SHARED_LINK_FLAGS_DEBUG )
set(libpng_EXE_LINK_FLAGS_DEBUG )
set(libpng_OBJECTS_DEBUG )
set(libpng_COMPILE_DEFINITIONS_DEBUG )
set(libpng_COMPILE_OPTIONS_C_DEBUG )
set(libpng_COMPILE_OPTIONS_CXX_DEBUG )
set(libpng_LIB_DIRS_DEBUG "${libpng_PACKAGE_FOLDER_DEBUG}/lib")
set(libpng_BIN_DIRS_DEBUG )
set(libpng_LIBRARY_TYPE_DEBUG STATIC)
set(libpng_IS_HOST_WINDOWS_DEBUG 1)
set(libpng_LIBS_DEBUG libpng16_staticd)
set(libpng_SYSTEM_LIBS_DEBUG )
set(libpng_FRAMEWORK_DIRS_DEBUG )
set(libpng_FRAMEWORKS_DEBUG )
set(libpng_BUILD_DIRS_DEBUG )
set(libpng_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(libpng_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${libpng_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${libpng_COMPILE_OPTIONS_C_DEBUG}>")
set(libpng_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libpng_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libpng_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libpng_EXE_LINK_FLAGS_DEBUG}>")


set(libpng_COMPONENTS_DEBUG )