"""Script is based on the cmake_example of pybind11 by Dean Moldovan: https://github.com/pybind/cmake_example"""
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.23.0':
                raise RuntimeError("CMake >= 3.23.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        # Allow extra cmake args (e.g. -DUSE_HIP=ON for AMD GPUs) via env.
        extra = os.environ.get('CMAKE_ARGS', '')
        if extra:
            cmake_args += extra.split()

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # A generator given in CMAKE_ARGS or CMAKE_GENERATOR (e.g. Ninja, which
        # the HIP build needs on Windows) does not accept the Visual Studio options.
        generator = os.environ.get('CMAKE_GENERATOR', '')
        for i, arg in enumerate(cmake_args):
            if arg.startswith('-G'):
                generator = arg[2:] or cmake_args[i + 1]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if not generator or generator.startswith('Visual Studio'):
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            elif 'Multi-Config' not in generator:
                cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='gpuRIR',
    ext_modules=[CMakeExtension('gpuRIR_bind')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
