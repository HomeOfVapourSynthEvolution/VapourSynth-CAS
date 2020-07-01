#ifdef CAS_X86
#include "CAS.h"

template<typename pixel_t>
void filter_avx2(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept {
    auto load_8u = [](const void * srcp) noexcept {
        if constexpr (std::is_same_v<pixel_t, uint8_t>)
            return Vec8i().load_8uc(srcp);
        else
            return Vec8i().load_8us(srcp);
    };

    auto store_8u = [&](const Vec8f __result, void * dstp) noexcept {
        const Vec8i _result = truncatei(__result + 0.5f);

        if constexpr (std::is_same_v<pixel_t, uint8_t>) {
            const auto result = compress_saturated_s2u(compress_saturated(_result, zero_si256()), zero_si256()).get_low();
            result.storel(dstp);
        } else {
            const auto result = compress_saturated_s2u(_result, zero_si256()).get_low();
            min(result, data->peak).store_nt(dstp);
        }
    };

    using var_t = std::conditional_t<std::is_integral_v<pixel_t>, Vec8i, Vec8f>;

    const var_t limit = std::any_cast<std::conditional_t<std::is_integral_v<pixel_t>, int, float>>(data->limit);

    auto filtering = [&](const var_t a, const var_t b, const var_t c, const var_t d, const var_t e, const var_t f, const var_t g, const var_t h, const var_t i,
                         const Vec8f chromaOffset) noexcept {
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
        Vec8f amp;
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
        const Vec8f weight = amp * data->sharpness;
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

            const Vec8f chromaOffset = plane ? 1.0f : 0.0f;

            const int regularPart = (width - 1) & ~(Vec8i().size() - 1);

            for (int y = 0; y < height; y++) {
                const pixel_t * above = srcp + (y == 0 ? stride : -stride);
                const pixel_t * below = srcp + (y == height - 1 ? -stride : stride);

                if constexpr (std::is_integral_v<pixel_t>) {
                    {
                        const Vec8i b = load_8u(above + 0);
                        const Vec8i e = load_8u(srcp + 0);
                        const Vec8i h = load_8u(below + 0);

                        const Vec8i a = permute8<1, 0, 1, 2, 3, 4, 5, 6>(b);
                        const Vec8i d = permute8<1, 0, 1, 2, 3, 4, 5, 6>(e);
                        const Vec8i g = permute8<1, 0, 1, 2, 3, 4, 5, 6>(h);

                        Vec8i c, f, i;
                        if (width > Vec8i().size()) {
                            c = load_8u(above + 1);
                            f = load_8u(srcp + 1);
                            i = load_8u(below + 1);
                        } else {
                            c = permute8<1, 2, 3, 4, 5, 6, 7, 6>(b);
                            f = permute8<1, 2, 3, 4, 5, 6, 7, 6>(e);
                            i = permute8<1, 2, 3, 4, 5, 6, 7, 6>(h);
                        }

                        const Vec8f result = filtering(a, b, c,
                                                       d, e, f,
                                                       g, h, i,
                                                       chromaOffset);

                        store_8u(result, dstp + 0);
                    }

                    for (int x = Vec8i().size(); x < regularPart; x += Vec8i().size()) {
                        const Vec8f result = filtering(load_8u(above + x - 1), load_8u(above + x), load_8u(above + x + 1),
                                                       load_8u(srcp + x - 1), load_8u(srcp + x), load_8u(srcp + x + 1),
                                                       load_8u(below + x - 1), load_8u(below + x), load_8u(below + x + 1),
                                                       chromaOffset);

                        store_8u(result, dstp + x);
                    }

                    if (regularPart >= Vec8i().size()) {
                        const Vec8i a = load_8u(above + regularPart - 1);
                        const Vec8i d = load_8u(srcp + regularPart - 1);
                        const Vec8i g = load_8u(below + regularPart - 1);

                        const Vec8i b = load_8u(above + regularPart);
                        const Vec8i e = load_8u(srcp + regularPart);
                        const Vec8i h = load_8u(below + regularPart);

                        const Vec8i c = permute8<1, 2, 3, 4, 5, 6, 7, 6>(b);
                        const Vec8i f = permute8<1, 2, 3, 4, 5, 6, 7, 6>(e);
                        const Vec8i i = permute8<1, 2, 3, 4, 5, 6, 7, 6>(h);

                        const Vec8f result = filtering(a, b, c,
                                                       d, e, f,
                                                       g, h, i,
                                                       chromaOffset);

                        store_8u(result, dstp + regularPart);
                    }
                } else {
                    {
                        const Vec8f b = Vec8f().load_a(above + 0);
                        const Vec8f e = Vec8f().load_a(srcp + 0);
                        const Vec8f h = Vec8f().load_a(below + 0);

                        const Vec8f a = permute8<1, 0, 1, 2, 3, 4, 5, 6>(b);
                        const Vec8f d = permute8<1, 0, 1, 2, 3, 4, 5, 6>(e);
                        const Vec8f g = permute8<1, 0, 1, 2, 3, 4, 5, 6>(h);

                        Vec8f c, f, i;
                        if (width > Vec8f().size()) {
                            c = Vec8f().load(above + 1);
                            f = Vec8f().load(srcp + 1);
                            i = Vec8f().load(below + 1);
                        } else {
                            c = permute8<1, 2, 3, 4, 5, 6, 7, 6>(b);
                            f = permute8<1, 2, 3, 4, 5, 6, 7, 6>(e);
                            i = permute8<1, 2, 3, 4, 5, 6, 7, 6>(h);
                        }

                        const Vec8f result = filtering(a, b, c,
                                                       d, e, f,
                                                       g, h, i,
                                                       chromaOffset);

                        result.store_nt(dstp + 0);
                    }

                    for (int x = Vec8f().size(); x < regularPart; x += Vec8f().size()) {
                        const Vec8f result = filtering(Vec8f().load(above + x - 1), Vec8f().load_a(above + x), Vec8f().load(above + x + 1),
                                                       Vec8f().load(srcp + x - 1), Vec8f().load_a(srcp + x), Vec8f().load(srcp + x + 1),
                                                       Vec8f().load(below + x - 1), Vec8f().load_a(below + x), Vec8f().load(below + x + 1),
                                                       chromaOffset);

                        result.store_nt(dstp + x);
                    }

                    if (regularPart >= Vec8f().size()) {
                        const Vec8f a = Vec8f().load(above + regularPart - 1);
                        const Vec8f d = Vec8f().load(srcp + regularPart - 1);
                        const Vec8f g = Vec8f().load(below + regularPart - 1);

                        const Vec8f b = Vec8f().load_a(above + regularPart);
                        const Vec8f e = Vec8f().load_a(srcp + regularPart);
                        const Vec8f h = Vec8f().load_a(below + regularPart);

                        const Vec8f c = permute8<1, 2, 3, 4, 5, 6, 7, 6>(b);
                        const Vec8f f = permute8<1, 2, 3, 4, 5, 6, 7, 6>(e);
                        const Vec8f i = permute8<1, 2, 3, 4, 5, 6, 7, 6>(h);

                        const Vec8f result = filtering(a, b, c,
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

template void filter_avx2<uint8_t>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
template void filter_avx2<uint16_t>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
template void filter_avx2<float>(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept;
#endif
