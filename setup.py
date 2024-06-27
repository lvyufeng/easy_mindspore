"""
setup packpage
"""
import os
import stat
import shlex
import shutil
import subprocess
from setuptools import find_packages
from setuptools import setup
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py


version = '0.0.1'
cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')

def clean():
    # pylint: disable=unused-argument
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    if os.path.exists(os.path.join(cur_dir, 'build')):
        shutil.rmtree(os.path.join(cur_dir, 'build'), onerror=readonly_handler)
    if os.path.exists(os.path.join(cur_dir, 'easy_mindspore.egg-info')):
        shutil.rmtree(os.path.join(cur_dir, 'easy_mindspore.egg-info'), onerror=readonly_handler)


clean()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    cmd = "git log --format='[sha1]:%h, [branch]:%d' -1"
    process = subprocess.Popen(
        shlex.split(cmd),
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    if not process.returncode:
        git_version = stdout.decode().strip()
        return "A torch-like frontend for MindSpore. Git version: %s" % (git_version)
    return "A torch-like frontend for MindSpore."


class EggInfo(egg_info):
    """Egg info."""
    def run(self):
        super().run()
        egg_info_dir = os.path.join(cur_dir, 'easy_mindspore.egg-info')
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""
    def run(self):
        super().run()
        mindarmour_dir = os.path.join(pkg_dir, 'lib', 'easy_mindspore')
        update_permissions(mindarmour_dir)


setup(
    name="easy_mindspore",
    version=version,
    author="lvyufeng",
    url="https://github.com/lvyufeng/easy_mindspore",
    project_urls={
        'Sources': 'https://github.com/lvyufeng/easy_mindspore',
        'Issue Tracker': 'https://github.com/lvyufeng/easy_mindspore/issues',
    },
    description=get_description(),
    license='MIT License',
    packages=find_packages(exclude=("example")),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    cmdclass={
        'egg_info': EggInfo,
        'build_py': BuildPy,
    },
    install_requires=[],
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ]
)
print(find_packages())