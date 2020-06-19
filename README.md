Description
===========

[Contrast Adaptive Sharpening](https://gpuopen.com/fidelityfx-cas/).


Usage
=====

    cas.CAS(clip clip[, float sharpness=0.0, int planes=[0, 1, 2], int opt=0])

* clip: Clip to process. Any planar format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.

* sharpness: Sharpening strength.

* planes: Sets which planes will be processed. Any unprocessed planes will be simply copied.

* opt: Sets which cpu optimizations to use.
  * 0 = auto detect
  * 1 = use c
  * 2 = use sse2
  * 3 = use avx2
  * 4 = use avx512


Compilation
===========

```
meson build
ninja -C build
ninja -C build install
```
