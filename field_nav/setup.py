from setuptools import setup, find_packages
import os
from glob import glob

package_name = "field_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"),
            glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "ekf_localizer         = field_nav.nodes.ekf_localizer:main",
            "crop_row_detector     = field_nav.nodes.crop_row_detector:main",
            "row_following_planner = field_nav.nodes.path_planner:main",
        ],
    },
)
