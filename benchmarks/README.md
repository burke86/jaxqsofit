# jaxqsofit Benchmarks

These scripts are manual diagnostics for profiling and pull-request checks. They
are not unit tests, and the exact runtimes depend on the local JAX backend,
hardware, and installed dependency versions.

## Core likelihood profiler

Profile the compiled log-density and value+gradient paths:

```bash
python benchmarks/jaxqsofit_likelihood_profile.py --host-sfh both --n-wave 3690
```

Useful comparison switches:

```bash
python benchmarks/jaxqsofit_likelihood_profile.py --host-sfh delayed --no-cache
python benchmarks/jaxqsofit_likelihood_profile.py --host-sfh flexible --convolution direct
```

The profiler uses a synthetic rest-frame spectrum, excludes JIT compilation from
the reported timings, and prints the active line count, line-group dimensions,
host template matrix shape, parameter count, forward time, and value+gradient
time for several likelihood variants.
