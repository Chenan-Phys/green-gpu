
set(GREEN_SYMMETRY_REPOSITORY "https://github.com/Chenan-Phys/green-symmetry.git" CACHE STRING "Coordinated THC host provider")
set(GREEN_SYMMETRY_REVISION "5d746d2a57c9d19f561497937fa493b7121d0b2a" CACHE STRING "Validated THC host provider revision")
function(add_green_dependency TARGET)
    Include(FetchContent)

    if(TARGET STREQUAL "green-symmetry")
      FetchContent_Declare(${TARGET} GIT_REPOSITORY ${GREEN_SYMMETRY_REPOSITORY} GIT_TAG ${GREEN_SYMMETRY_REVISION})
    else()
    FetchContent_Declare(
        ${TARGET}
        GIT_REPOSITORY https://github.com/Green-Phys/${TARGET}.git
        GIT_TAG  ${GREEN_RELEASE} # or a later release
    )
    endif()

    FetchContent_MakeAvailable(${TARGET})
endfunction()
