# sparselu
LU decomposition of sparse matrices in Rcpp using UMFPACK.

## Installation

The package links against the SuiteSparse UMFPACK libraries. Install SuiteSparse (and its dependencies such as AMD and SuiteSparse_config) prior to building the package. For example:

- macOS (Homebrew): `brew install suite-sparse`
- Debian/Ubuntu: `sudo apt install libsuitesparse-dev`

If SuiteSparse is installed in a non-standard location, set the environment variable `SUITESPARSE_HOME`, or provide `PKG_CPPFLAGS`/`PKG_LIBS` when installing the package so that the headers and libraries can be located.
