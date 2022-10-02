# PROPACK

## Overview

This custom PROPACK repository is simply a fork of the original [PROPACK77
library](http://sun.stanford.edu/~rmunk/PROPACK/). It adds the following
modifications to the original package:
- Move from CVS to git.
- Introduce the CMake build system.
- In `[double|single|complex16|complex8]/second.F`, `ETIME` is no longer
  declared as external.
- Remove `*.ascii` output files.
- Adds `scripts` folder for testing.

## Installation

Extract the files using
```
tar -xzf PROPACK.tar.gz
```

Change into the PROPACK directory and create a build workspace
```
cd PROPACK
mkdir build
cd build
```

Configure using `cmake`
```
cmake ..
```

Note: PROPACK requires BLAS and LAPACK implementation, make sure these
libraries exist in your system before invoking cmake.

Build and install with `make`
```
make -j N  # N is the number of threads, e.g., 4
make install
```

By default, PROPACK will be installed into `/usr/local`. Make sure you
have the write permission to that folder, or you can pass `-DCMAKE_INSTALL_PREFIX`
to `cmake` to install into another location.

Optionally, use `ctest` to run several PROPACK tests
```
ctest
```

## Copyright
I don't own anything. PROPACK was written by Rasmus Munk Larsen (rmunk@quake.stanford.edu).
See `README` for more details.
