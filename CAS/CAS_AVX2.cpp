#ifdef CAS_X86
#include "CAS.h"

template<typename pixel_t>
void filter_avx2(const VSFrameRef * src, VSFrameRef * dst, const CASData * const VS_RESTRICT data, const VSAPI * vsapi) noexcept {
    using var_t = std::conditional_t<std::is_integral_v<pixel_t>, int, float>;
    using vec_t = std::conditional_t<std::is_integral_v<pixel_t>, Vec8i, Vec8f>;

    const vec_t limit = std::any_cast<var_t>(data->limit);

    auto load = [](const pixel_t * srcp) noexcept {
        if constexpr (std::is_same_v<pixel_t, uint8_t>)
            return vec_t().load_8uc(srcp);
        else if constexpr (std::is_same_v<pixel_t, uint16_t>)
            return vec_t().load_8us(srcp);
        else
            return vec_t().load(srcp);
    };

    auto store = [&](const Vec8f srcp, pixel_t * dstp) noexcept {
        if constexpr (std::is_same_v<pixel_t, uint8_t>) {
            const auto result = compress_saturated_s2u(compress_saturated(truncatei(srcp + 0.5f), zero_si256()), zero_si256()).get_low();
            result.storel(dstp);
        } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
            const auto result = compress_saturated_s2u(truncatei(srcp + 0.5f), zero_si256()).get_low();
            min(result, data->peak).store_nt(dstp);
        } else {
            srcp.store_nt(dstp);
        }
    };

    auto filtering = [&](const vec_t a, const vec_t b, const vec_t c, const vec_t d, const vec_t e, const vec_t f, const vec_t g, const vec_t h, const vec_t i,
                         const Vec8f chromaOffset) noexcept {
        // Soft min and max.
        //  a b c             b
        //  d e f * 0.5  +  d e f * 0.5
        //  g h i             h
        // These are 2.0x bigger (factored out the extra multiply).
        vec_t mn = min(min(min(d, e), min(f, b)), h);
        const vec_t mn2 = min(min(min(mn, a), min(c, g)), i);
        mn += mn2;

        vec_t mx = max(max(max(d, e), max(f, b)), h);
        const vec_t mx2 = max(max(max(mx, a), max(c, g)), i);
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

            const int regularPart = (width - 1) & ~(vec_t().size() - 1);

            for (int y = 0; y < height; y++) {
                const pixel_t * above = srcp + (y == 0 ? stride : -stride);
                const pixel_t * below = srcp + (y == height - 1 ? -stride : stride);

                {
                    const vec_t b = load(above + 0);
                    const vec_t e = load(srcp + 0);
                    const vec_t h = load(below + 0);

                    const vec_t a = permute8<1, 0, 1, 2, 3, 4, 5, 6>(b);
                    const vec_t d = permute8<1, 0, 1, 2, 3, 4, 5, 6>(e);
                    const vec_t g = permute8<1, 0, 1, 2, 3, 4, 5, 6>(h);

                    vec_t c, f, i;
                    if (width > vec_t().size()) {
                        c = load(above + 1);
                        f = load(srcp + 1);
                        i = load(below + 1);
                    } else {
                        c = permute8<1, 2, 3, 4, 5, 6, 7, 6>(b);
                        f = permute8<1, 2, 3, 4, 5, 6, 7, 6>(e);
                        i = permute8<1, 2, 3, 4, 5, 6, 7, 6>(h);
                    }

                    const Vec8f result = filtering(a, b, c,
                                                   d, e, f,
                                                   g, h, i,
                                                   chromaOffset);

                    store(result, dstp + 0);
                }

                for (int x = vec_t().size(); x < regularPart; x += vec_t().size()) {
                    const Vec8f result = filtering(load(above + x - 1), load(above + x), load(above + x + 1),
                                                   load(srcp + x - 1), load(srcp + x), load(srcp + x + 1),
                                                   load(below + x - 1), load(below + x), load(below + x + 1),
                                                   chromaOffset);

                    store(result, dstp + x);
                }

                if (regularPart >= vec_t().size()) {
                    const vec_t a = load(above + regularPart - 1);
                    const vec_t d = load(srcp + regularPart - 1);
                    const vec_t g = load(below + regularPart - 1);

                    const vec_t b = load(above + regularPart);
                    const vec_t e = load(srcp + regularPart);
                    const vec_t h = load(below + regularPart);

                    const vec_t c = permute8<1, 2, 3, 4, 5, 6, 7, 6>(b);
                    const vec_t f = permute8<1, 2, 3, 4, 5, 6, 7, 6>(e);
                    const vec_t i = permute8<1, 2, 3, 4, 5, 6, 7, 6>(h);

                    const Vec8f result = filtering(a, b, c,
                                                   d, e, f,
                                                   g, h, i,
                                                   chromaOffset);

                    store(result, dstp + regularPart);
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
