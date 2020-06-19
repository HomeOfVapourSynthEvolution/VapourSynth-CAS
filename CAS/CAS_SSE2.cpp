#ifdef CAS_X86
#include "CAS.h"

template<typename pixel_t>
void filter_sse2(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept {
    auto load_4u = [](const void * srcp) noexcept {
        if constexpr (std::is_same_v<pixel_t, uint8_t>)
            return Vec4i().load_4uc(srcp);
        else
            return Vec4i().load_4us(srcp);
    };

    auto store_4u = [=](const Vec4f __result, void * dstp) noexcept {
        const Vec4i _result = truncatei(__result + 0.5f);

        if constexpr (std::is_same_v<pixel_t, uint8_t>) {
            const auto result = compress_saturated_s2u(compress_saturated(_result, zero_si128()), zero_si128());
            result.store_si32(dstp);
        } else {
            const auto result = compress_saturated_s2u(_result, zero_si128());
            min(result, data->peak).storel(dstp);
        }
    };

    using var_t = std::conditional_t<std::is_integral_v<pixel_t>, Vec4i, Vec4f>;

    const var_t limit = std::any_cast<std::conditional_t<std::is_integral_v<pixel_t>, int, float>>(data->limit);

    auto filtering = [=](const var_t a, const var_t b, const var_t c, const var_t d, const var_t e, const var_t f, const var_t g, const var_t h, const var_t i,
                         const Vec4f chromaOffset) noexcept {
        // Soft min and max.
        //  a b c             b
        //  d e f * 0.5  +  d e f * 0.5
        //  g h i             h
        // These are 2.0x bigger (factored out the extra multiply).
        var_t mn = min(min(min(d, e), min(f, b)), h);
        const var_t mn2 = min(min(min(mn, a), min(c, g)), i);
        mn += mn2;

        var_t mx = max(max(max(d, e), max(f, b)), h);
        const var_t mx2 = max(max(max(mx, a), max(c, g)), i);
        mx += mx2;

        if constexpr (std::is_floating_point_v<pixel_t>) {
            mn += chromaOffset;
            mx += chromaOffset;
        }

        // Smooth minimum distance to signal limit divided by smooth max.
        Vec4f amp;
        if constexpr (std::is_integral_v<pixel_t>)
            amp = min(max(to_float(min(mn, limit - mx)) / to_float(mx), 0.0f), 1.0f);
        else
            amp = min(max(min(mn, limit - mx) / mx, 0.0f), 1.0f);

        // Shaping amount of sharpening.
        amp = sqrt(amp);

        // Filter shape.
        //  0 w 0
        //  w 1 w
        //  0 w 0
        const Vec4f weight = amp * data->sharpness;
        if constexpr (std::is_integral_v<pixel_t>)
            return mul_add(to_float((b + d) + (f + h)), weight, to_float(e)) / mul_add(4.0f, weight, 1.0f);
        else
            return mul_add((b + d) + (f + h), weight, e) / mul_add(4.0f, weight, 1.0f);
    };

    for (int plane = 0; plane < data->vi->format->numPlanes; plane++) {
        if (data->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const int stride = vsapi->getStride(src, plane) / sizeof(pixel_t);
            const pixel_t * srcp = reinterpret_cast<const pixel_t *>(vsapi->getReadPtr(src, plane));
            pixel_t * dstp = reinterpret_cast<pixel_t *>(vsapi->getWritePtr(dst, plane));

            const Vec4f chromaOffset = plane ? 1.0f : 0.0f;

            const int regularPart = (width - 1) & ~(Vec4i().size() - 1);

            for (int y = 0; y < height; y++) {
                const pixel_t * above = srcp + (y == 0 ? stride : -stride);
                const pixel_t * below = srcp + (y == height - 1 ? -stride : stride);

                if constexpr (std::is_integral_v<pixel_t>) {
                    {
                        const Vec4i b = load_4u(above + 0);
                        const Vec4i e = load_4u(srcp + 0);
                        const Vec4i h = load_4u(below + 0);

                        const Vec4i a = permute4<1, 0, 1, 2>(b);
                        const Vec4i d = permute4<1, 0, 1, 2>(e);
                        const Vec4i g = permute4<1, 0, 1, 2>(h);

                        Vec4i c, f, i;
                        if (width > Vec4i().size()) {
                            c = load_4u(above + 1);
                            f = load_4u(srcp + 1);
                            i = load_4u(below + 1);
                        } else {
                            c = permute4<1, 2, 3, 2>(b);
                            f = permute4<1, 2, 3, 2>(e);
                            i = permute4<1, 2, 3, 2>(h);
                        }

                        const Vec4f result = filtering(a, b, c,
                                                       d, e, f,
                                                       g, h, i,
                                                       chromaOffset);

                        store_4u(result, dstp + 0);
                    }

                    for (int x = Vec4i().size(); x < regularPart; x += Vec4i().size()) {
                        const Vec4f result = filtering(load_4u(above + x - 1), load_4u(above + x), load_4u(above + x + 1),
                                                       load_4u(srcp + x - 1), load_4u(srcp + x), load_4u(srcp + x + 1),
                                                       load_4u(below + x - 1), load_4u(below + x), load_4u(below + x + 1),
                                                       chromaOffset);

                        store_4u(result, dstp + x);
                    }

                    if (regularPart >= Vec4i().size()) {
                        const Vec4i a = load_4u(above + regularPart - 1);
                        const Vec4i d = load_4u(srcp + regularPart - 1);
                        const Vec4i g = load_4u(below + regularPart - 1);

                        const Vec4i b = load_4u(above + regularPart);
                        const Vec4i e = load_4u(srcp + regularPart);
                        const Vec4i h = load_4u(below + regularPart);

                        const Vec4i c = permute4<1, 2, 3, 2>(b);
                        const Vec4i f = permute4<1, 2, 3, 2>(e);
                        const Vec4i i = permute4<1, 2, 3, 2>(h);

                        const Vec4f result = filtering(a, b, c,
                                                       d, e, f,
                                                       g, h, i,
                                                       chromaOffset);

                        store_4u(result, dstp + regularPart);
                    }
                } else {
                    {
                        const Vec4f b = Vec4f().load_a(above + 0);
                        const Vec4f e = Vec4f().load_a(srcp + 0);
                        const Vec4f h = Vec4f().load_a(below + 0);

                        const Vec4f a = permute4<1, 0, 1, 2>(b);
                        const Vec4f d = permute4<1, 0, 1, 2>(e);
                        const Vec4f g = permute4<1, 0, 1, 2>(h);

                        Vec4f c, f, i;
                        if (width > Vec4f().size()) {
                            c = Vec4f().load(above + 1);
                            f = Vec4f().load(srcp + 1);
                            i = Vec4f().load(below + 1);
                        } else {
                            c = permute4<1, 2, 3, 2>(b);
                            f = permute4<1, 2, 3, 2>(e);
                            i = permute4<1, 2, 3, 2>(h);
                        }

                        const Vec4f result = filtering(a, b, c,
                                                       d, e, f,
                                                       g, h, i,
                                                       chromaOffset);

                        result.store_nt(dstp + 0);
                    }

                    for (int x = Vec4f().size(); x < regularPart; x += Vec4f().size()) {
                        const Vec4f result = filtering(Vec4f().load(above + x - 1), Vec4f().load_a(above + x), Vec4f().load(above + x + 1),
                                                       Vec4f().load(srcp + x - 1), Vec4f().load_a(srcp + x), Vec4f().load(srcp + x + 1),
                                                       Vec4f().load(below + x - 1), Vec4f().load_a(below + x), Vec4f().load(below + x + 1),
                                                       chromaOffset);

                        result.store_nt(dstp + x);
                    }

                    if (regularPart >= Vec4f().size()) {
                        const Vec4f a = Vec4f().load(above + regularPart - 1);
                        const Vec4f d = Vec4f().load(srcp + regularPart - 1);
                        const Vec4f g = Vec4f().load(below + regularPart - 1);

                        const Vec4f b = Vec4f().load_a(above + regularPart);
                        const Vec4f e = Vec4f().load_a(srcp + regularPart);
                        const Vec4f h = Vec4f().load_a(below + regularPart);

                        const Vec4f c = permute4<1, 2, 3, 2>(b);
                        const Vec4f f = permute4<1, 2, 3, 2>(e);
                        const Vec4f i = permute4<1, 2, 3, 2>(h);

                        const Vec4f result = filtering(a, b, c,
                                                       d, e, f,
                                                       g, h, i,
                                                       chromaOffset);

                        result.store_nt(dstp + regularPart);
                    }
                }

                srcp += stride;
                dstp += stride;
            }
        }
    }
}

template void filter_sse2<uint8_t>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
template void filter_sse2<uint16_t>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
template void filter_sse2<float>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
#endif
