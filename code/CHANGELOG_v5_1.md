# v5.1 build fix

- Creates the Windows `build` directory before invoking Cygwin.
- Compiles all intermediate files and executables in Cygwin `/tmp`.
- Copies only verified executables to the project `build` directory.
- Avoids creating `.filesystem_test.cpp` directly on the Windows project volume.
