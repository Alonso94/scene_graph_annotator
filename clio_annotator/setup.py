from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['clio_aeg_annotation'],
    package_dir={'': 'src'},
    requires=['rospy', 'std_msgs', 'geometry_msgs', 'hydra_msgs']
)

setup(**setup_args)