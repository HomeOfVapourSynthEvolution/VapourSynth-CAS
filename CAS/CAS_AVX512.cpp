#ifdef CAS_X86
#include "CAS.h"

template<typename pixel_t>
void filter_avx512(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept {
    auto load_16u = [](const void * srcp) noexcept {
        if constexpr (std::is_same_v<pixel_t, uint8_t>)
            return Vec16i().load_16uc(srcp);
        else
            return Vec16i().load_16us(srcp);
    };

    auto store_16u = [=](const Vec16f __result, void * dstp) noexcept {
        const Vec16i _result = truncatei(__result + 0.5f);

        if constexpr (std::is_same_v<pixel_t, uint8_t>) {
            const auto result = compress_saturated_s2u(compress_saturated(_result, zero_si512()), zero_si512()).get_low().get_low();
            result.store_nt(dstp);
        } else {
            const auto result = compress_saturated_s2u(_result, zero_si512()).get_low();
            min(result, data->peak).store_nt(dstp);
        }
    };

    using var_t = std::conditional_t<std::is_integral_v<pixel_t>, Vec16i, Vec16f>;

    const var_t limit = std::any_cast<std::conditional_t<std::is_integral_v<pixel_t>, int, float>>(data->limit);

    auto filtering = [=](const var_t a, const var_t b, const var_t c, const var_t d, const var_t e, const var_t f, const var_t g, const var_t h, const var_t i,
                         const Vec16f chromaOffset) noexcept {
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
        Vec16f amp;
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
        const Vec16f weight = amp * data->sharpness;
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

            const Vec16f chromaOffset = plane ? 1.0f : 0.0f;

            const int regularPart = (width - 1) & ~(Vec16i().size() - 1);

            for (int y = 0; y < height; y++) {
                const pixel_t * above = srcp + (y == 0 ? stride : -stride);
                const pixel_t * below = srcp + (y == height - 1 ? -stride : stride);

                if constexpr (std::is_integral_v<pixel_t>) {
                    {
                        const Vec16i b = load_16u(above + 0);
                        const Vec16i e = load_16u(srcp + 0);
                        const Vec16i h = load_16u(below + 0);

                        const Vec16i a = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(b);
                        const Vec16i d = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(e);
                        const Vec16i g = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(h);

                        Vec16i c, f, i;
                        if (width > Vec16i().size()) {
                            c = load_16u(above + 1);
                            f = load_16u(srcp + 1);
                            i = load_16u(below + 1);
                        } else {
                            c = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(b);
                            f = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(e);
                            i = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(h);
                        }

                        const Vec16f result = filtering(a, b, c,
                                                        d, e, f,
                                                        g, h, i,
                                                        chromaOffset);

                        store_16u(result, dstp + 0);
                    }

                    for (int x = Vec16i().size(); x < regularPart; x += Vec16i().size()) {
                        const Vec16f result = filtering(load_16u(above + x - 1), load_16u(above + x), load_16u(above + x + 1),
                                                        load_16u(srcp + x - 1), load_16u(srcp + x), load_16u(srcp + x + 1),
                                                        load_16u(below + x - 1), load_16u(below + x), load_16u(below + x + 1),
                                                        chromaOffset);

                        store_16u(result, dstp + x);
                    }

                    if (regularPart >= Vec16i().size()) {
                        const Vec16i a = load_16u(above + regularPart - 1);
                        const Vec16i d = load_16u(srcp + regularPart - 1);
                        const Vec16i g = load_16u(below + regularPart - 1);

                        const Vec16i b = load_16u(above + regularPart);
                        const Vec16i e = load_16u(srcp + regularPart);
                        const Vec16i h = load_16u(below + regularPart);

                        const Vec16i c = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(b);
                        const Vec16i f = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(e);
                        const Vec16i i = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(h);

                        const Vec16f result = filtering(a, b, c,
                                                        d, e, f,
                                                        g, h, i,
                                                        chromaOffset);

                        store_16u(result, dstp + regularPart);
                    }
                } else {
                    {
                        const Vec16f b = Vec16f().load_a(above + 0);
                        const Vec16f e = Vec16f().load_a(srcp + 0);
                        const Vec16f h = Vec16f().load_a(below + 0);

                        const Vec16f a = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(b);
                        const Vec16f d = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(e);
                        const Vec16f g = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(h);

                        Vec16f c, f, i;
                        if (width > Vec16f().size()) {
                            c = Vec16f().load(above + 1);
                            f = Vec16f().load(srcp + 1);
                            i = Vec16f().load(below + 1);
                        } else {
                            c = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(b);
                            f = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(e);
                            i = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(h);
                        }

                        const Vec16f result = filtering(a, b, c,
                                                        d, e, f,
                                                        g, h, i,
                                                        chromaOffset);

                        result.store_nt(dstp + 0);
                    }

                    for (int x = Vec16f().size(); x < regularPart; x += Vec16f().size()) {
                        const Vec16f result = filtering(Vec16f().load(above + x - 1), Vec16f().load_a(above + x), Vec16f().load(above + x + 1),
                                                        Vec16f().load(srcp + x - 1), Vec16f().load_a(srcp + x), Vec16f().load(srcp + x + 1),
                                                        Vec16f().load(below + x - 1), Vec16f().load_a(below + x), Vec16f().load(below + x + 1),
                                                        chromaOffset);

                        result.store_nt(dstp + x);
                    }

                    if (regularPart >= Vec16f().size()) {
                        const Vec16f a = Vec16f().load(above + regularPart - 1);
                        const Vec16f d = Vec16f().load(srcp + regularPart - 1);
                        const Vec16f g = Vec16f().load(below + regularPart - 1);

                        const Vec16f b = Vec16f().load_a(above + regularPart);
                        const Vec16f e = Vec16f().load_a(srcp + regularPart);
                        const Vec16f h = Vec16f().load_a(below + regularPart);

                        const Vec16f c = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(b);
                        const Vec16f f = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(e);
                        const Vec16f i = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(h);

                        const Vec16f result = filtering(a, b, c,
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

template void filter_avx512<uint8_t>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
template void filter_avx512<uint16_t>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
template void filter_avx512<float>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
#endif
