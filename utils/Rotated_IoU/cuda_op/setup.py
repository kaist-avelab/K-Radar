from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sort_vertices',
    ext_modules=[
        CUDAExtension('sort_vertices', [
            'sort_vert.cpp',
            'sort_vert_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })