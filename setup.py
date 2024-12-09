from setuptools import setup

setup(name = "amescap",
      version = "0.3",
      description = "Analysis pipeline for the NASA Ames MGCM",
      url = "https://github.com/NASA-Planetary-Science/AmesCAP",
      author = "Mars Climate Modeling Center",
      author_email = "alexandre.m.kling@nasa.gov",
      license = "MIT License",
      scripts = ["bin/MarsPull.py", "bin/MarsInterp.py", 
                 "bin/MarsPlot.py", "bin/MarsVars.py", 
                 "bin/MarsFiles.py", "bin/MarsFormat.py", 
                 "bin/MarsCalendar.py"],
      install_requires = ["requests", "netCDF4", "numpy", "matplotlib", 
                          "scipy", "xarray"],
      packages = ["amescap"],
      data_files = [
            ("mars_data", ["mars_data/Legacy.fixed.nc"]),
            ("mars_templates", ["mars_templates/legacy.in", 
                                "mars_templates/amescap_profile"])
            ],
      include_package_data = True,
      zip_safe = False)
