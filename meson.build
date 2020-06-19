project('CAS', 'cpp',
  default_options: ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++17'],
  meson_version: '>=0.48.0',
  version: '1'
)

sources = [
  'CAS/CAS.cpp',
  'CAS/CAS.h'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DCAS_X86', '-mfpmath=sse', '-msse2', language: 'cpp')

  sources += [
    'CAS/CAS_SSE2.cpp',
    'CAS/VCL2/instrset.h',
    'CAS/VCL2/instrset_detect.cpp',
    'CAS/VCL2/vector_convert.h',
    'CAS/VCL2/vectorclass.h',
    'CAS/VCL2/vectorf128.h',
    'CAS/VCL2/vectorf256.h',
    'CAS/VCL2/vectorf256e.h',
    'CAS/VCL2/vectorf512.h',
    'CAS/VCL2/vectorf512e.h',
    'CAS/VCL2/vectori128.h',
    'CAS/VCL2/vectori256.h',
    'CAS/VCL2/vectori256e.h',
    'CAS/VCL2/vectori512.h',
    'CAS/VCL2/vectori512e.h',
    'CAS/VCL2/vectori512s.h',
    'CAS/VCL2/vectori512se.h',
    'CAS/VCL2/vectormath_common.h',
    'CAS/VCL2/vectormath_exp.h',
    'CAS/VCL2/vectormath_hyp.h',
    'CAS/VCL2/vectormath_lib.h',
    'CAS/VCL2/vectormath_trig.h'
  ]

  libs += static_library('avx2', 'CAS/CAS_AVX2.cpp',
    dependencies: vapoursynth_dep,
    cpp_args: ['-mavx2', '-mfma'],
    gnu_symbol_visibility: 'hidden'
  )

  libs += static_library('avx512', 'CAS/CAS_AVX512.cpp',
    dependencies: vapoursynth_dep,
    cpp_args: ['-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl', '-mfma'],
    gnu_symbol_visibility: 'hidden'
  )
endif

shared_module('cas', sources,
  dependencies: vapoursynth_dep,
  link_with: libs,
  install: true,
  install_dir: join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth'),
  gnu_symbol_visibility: 'hidden'
)
