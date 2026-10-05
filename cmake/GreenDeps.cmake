
function(add_green_dependency TARGET)
    Include(FetchContent)

    set(dependency_repository https://github.com/Green-Phys/${TARGET}.git)
    set(dependency_revision ${GREEN_RELEASE})
    if("${TARGET}" STREQUAL "green-symmetry")
        set(GREEN_SYMMETRY_GIT_REPOSITORY "https://github.com/Chenan-Phys/green-symmetry.git" CACHE STRING "SG reader dependency repository")
        set(GREEN_SYMMETRY_GIT_TAG "82b5c67a5c2f6d53dd0fbf80178eb1a234879041" CACHE STRING "Validated SG reader dependency revision")
        set(dependency_repository ${GREEN_SYMMETRY_GIT_REPOSITORY})
        set(dependency_revision ${GREEN_SYMMETRY_GIT_TAG})
    endif()
    FetchContent_Declare(
        ${TARGET}
        GIT_REPOSITORY ${dependency_repository}
        GIT_TAG ${dependency_revision}
    )

    FetchContent_MakeAvailable(${TARGET})
endfunction()
