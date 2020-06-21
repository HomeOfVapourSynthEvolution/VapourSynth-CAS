/*
    MIT License

    Copyright (c) 2020 Holy Wu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <cmath>

#include <algorithm>
#include <memory>
#include <string>

#include "CAS.h"

#ifdef CAS_X86
template<typename pixel_t> extern void filter_sse2(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
template<typename pixel_t> extern void filter_avx2(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
template<typename pixel_t> extern void filter_avx512(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
#endif

template<typename pixel_t>
static void filter_c(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept {
    using var_t = std::conditional_t<std::is_integral_v<pixel_t>, int, float>;

    const var_t limit = std::any_cast<var_t>(data->limit);

    auto filtering = [=](const var_t a, const var_t b, const var_t c, const var_t d, const var_t e, const var_t f, const var_t g, const var_t h, const var_t i,
                         const float chromaOffset) noexcept {
        // Soft min and max.
        //  a b c             b
        //  d e f * 0.5  +  d e f * 0.5
        //  g h i             h
        // These are 2.0x bigger (factored out the extra multiply).
        var_t mn = std::min({ d, e, f, b, h });
        const var_t mn2 = std::min({ mn, a, c, g, i });
        mn += mn2;

        var_t mx = std::max({ d, e, f, b, h });
        const var_t mx2 = std::max({ mx, a, c, g, i });
        mx += mx2;

        if constexpr (std::is_floating_point_v<pixel_t>) {
            mn += chromaOffset;
            mx += chromaOffset;
        }

        // Smooth minimum distance to signal limit divided by smooth max.
        float amp = std::clamp(std::min(mn, limit - mx) / static_cast<float>(mx), 0.0f, 1.0f);

        // Shaping amount of sharpening.
        amp = std::sqrt(amp);

        // Filter shape.
        //  0 w 0
        //  w 1 w
        //  0 w 0
        const float weight = amp * data->sharpness;
        return ((b + d + f + h) * weight + e) / (1.0f + 4.0f * weight);
    };

    for (int plane = 0; plane < data->vi->format->numPlanes; plane++) {
        if (data->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const int stride = vsapi->getStride(src, plane) / sizeof(pixel_t);
            const pixel_t * srcp = reinterpret_cast<const pixel_t *>(vsapi->getReadPtr(src, plane));
            pixel_t * VS_RESTRICT dstp = reinterpret_cast<pixel_t *>(vsapi->getWritePtr(dst, plane));

            const float chromaOffset = plane ? 1.0f : 0.0f;

            for (int y = 0; y < height; y++) {
                const pixel_t * above = srcp + (y == 0 ? stride : -stride);
                const pixel_t * below = srcp + (y == height - 1 ? -stride : stride);

                {
                    const float result = filtering(above[1], above[0], above[1],
                                                   srcp[1], srcp[0], srcp[1],
                                                   below[1], below[0], below[1],
                                                   chromaOffset);

                    if constexpr (std::is_integral_v<pixel_t>)
                        dstp[0] = std::clamp(static_cast<int>(result + 0.5f), 0, data->peak);
                    else
                        dstp[0] = result;
                }

                for (int x = 1; x < width - 1; x++) {
                    const float result = filtering(above[x - 1], above[x], above[x + 1],
                                                   srcp[x - 1], srcp[x], srcp[x + 1],
                                                   below[x - 1], below[x], below[x + 1],
                                                   chromaOffset);

                    if constexpr (std::is_integral_v<pixel_t>)
                        dstp[x] = std::clamp(static_cast<int>(result + 0.5f), 0, data->peak);
                    else
                        dstp[x] = result;
                }

                {
                    const float result = filtering(above[width - 2], above[width - 1], above[width - 2],
                                                   srcp[width - 2], srcp[width - 1], srcp[width - 2],
                                                   below[width - 2], below[width - 1], below[width - 2],
                                                   chromaOffset);

                    if constexpr (std::is_integral_v<pixel_t>)
                        dstp[width - 1] = std::clamp(static_cast<int>(result + 0.5f), 0, data->peak);
                    else
                        dstp[width - 1] = result;
                }

                srcp += stride;
                dstp += stride;
            }
        }
    }
}

static void VS_CC casInit(VSMap * in, VSMap * out, void ** instanceData, VSNode * node, VSCore * core, const VSAPI * vsapi) {
    CASData * d = static_cast<CASData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef * VS_CC casGetFrame(int n, int activationReason, void ** instanceData, void ** frameData, VSFrameContext * frameCtx, VSCore * core, const VSAPI * vsapi) {
    const CASData * d = static_cast<const CASData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef * fr[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] = { 0, 1, 2 };
        VSFrameRef * dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        d->filter(src, dst, d, vsapi);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC casFree(void * instanceData, VSCore * core, const VSAPI * vsapi) {
    CASData * d = static_cast<CASData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC casCreate(const VSMap * in, VSMap * out, void * userData, VSCore * core, const VSAPI * vsapi) {
    using namespace std::literals;

    std::unique_ptr<CASData> d = std::make_unique<CASData>();

    try {
        d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);
        int err;

        if (!isConstantFormat(d->vi) ||
            (d->vi->format->sampleType == stInteger && d->vi->format->bitsPerSample > 16) ||
            (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported"sv;

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (d->vi->width >> (plane ? d->vi->format->subSamplingW : 0) < 3)
                throw "every plane's width must be greater than or equal to 3"sv;

            if (d->vi->height >> (plane ? d->vi->format->subSamplingH : 0) < 3)
                throw "every plane's height must be greater than or equal to 3"sv;
        }

        d->sharpness = static_cast<float>(vsapi->propGetFloat(in, "sharpness", 0, &err));

        {
            const int m = vsapi->propNumElements(in, "planes");

            if (m <= 0) {
                for (int i = 0; i < 3; i++) {
                    d->process[i] = true;
                    if (i == 0 && d->vi->format->colorFamily != cmRGB)
                        break;
                }
            }

            for (int i = 0; i < m; i++) {
                const int n = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));

                if (n < 0 || n >= d->vi->format->numPlanes)
                    throw "plane index out of range"sv;

                if (d->process[n])
                    throw "plane specified twice"sv;

                d->process[n] = true;
            }
        }

        const int opt = int64ToIntS(vsapi->propGetInt(in, "opt", 0, &err));

        if (d->sharpness < 0.0f || d->sharpness > 1.0f)
            throw "sharpness must be between 0.0 and 1.0 (inclusive)"sv;

        if (opt < 0 || opt > 4)
            throw "opt must be 0, 1, 2, 3, or 4"sv;

        {
            if (d->vi->format->bytesPerSample == 1)
                d->filter = filter_c<uint8_t>;
            else if (d->vi->format->bytesPerSample == 2)
                d->filter = filter_c<uint16_t>;
            else
                d->filter = filter_c<float>;

#ifdef CAS_X86
            const int iset = instrset_detect();
            if ((opt == 0 && iset >= 10) || opt == 4) {
                if (d->vi->format->bytesPerSample == 1)
                    d->filter = filter_avx512<uint8_t>;
                else if (d->vi->format->bytesPerSample == 2)
                    d->filter = filter_avx512<uint16_t>;
                else
                    d->filter = filter_avx512<float>;
            } else if ((opt == 0 && iset >= 8) || opt == 3) {
                if (d->vi->format->bytesPerSample == 1)
                    d->filter = filter_avx2<uint8_t>;
                else if (d->vi->format->bytesPerSample == 2)
                    d->filter = filter_avx2<uint16_t>;
                else
                    d->filter = filter_avx2<float>;
            } else if ((opt == 0 && iset >= 2) || opt == 2) {
                if (d->vi->format->bytesPerSample == 1)
                    d->filter = filter_sse2<uint8_t>;
                else if (d->vi->format->bytesPerSample == 2)
                    d->filter = filter_sse2<uint16_t>;
                else
                    d->filter = filter_sse2<float>;
            }
#endif
        }

        auto lerp = [](const float a, const float b, const float t) noexcept { return a + (b - a) * t; };
        d->sharpness = -1.0f / lerp(16.0f, 5.0f, d->sharpness);

        if (d->vi->format->sampleType == stInteger)
            d->limit = (1 << (d->vi->format->bitsPerSample + 1)) - 1;
        else
            d->limit = 2.0f;

        d->peak = (1 << d->vi->format->bitsPerSample) - 1;
    } catch (const std::string_view & error) {
        vsapi->setError(out, ("CAS: "s + error.data()).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    vsapi->createFilter(in, out, "CAS", casInit, casGetFrame, casFree, fmParallel, 0, d.release(), core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin * plugin) {
    configFunc("com.holywu.cas", "cas", "Contrast Adaptive Sharpening", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("CAS",
                 "clip:clip;"
                 "sharpness:float:opt;"
                 "planes:int[]:opt;"
                 "opt:int:opt;",
                 casCreate, nullptr, plugin);
}
