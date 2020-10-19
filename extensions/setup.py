from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer',
    ext_modules=[
        CUDAExtension('chamfer', [
            'chamfer_cuda.cpp',
            'chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='grid_pooling',
    ext_modules=[
        CUDAExtension('grid_pooling', [
            'grid_pooling_cuda.cpp',
            'grid_pooling.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
