#pragma once

#include <any>
#include <type_traits>

#include <VapourSynth.h>
#include <VSHelper.h>

#ifdef CAS_X86
#include "VCL2/vectorclass.h"
#endif

struct CASData final {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    float sharpness;
    bool process[3];
    std::any limit;
    int peak;
    void (*filter)(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
};
