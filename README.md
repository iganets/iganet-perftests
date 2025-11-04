# IGAnets-PerfTests

[![GitlabSync](https://github.com/iganets/iganet-perftests/actions/workflows/gitlab-sync.yml/badge.svg)](https://github.com/iganets/iganet-perftests/actions/workflows/gitlab-sync.yml)
[![CI](https://github.com/iganets/iganet-perftests/actions/workflows/ci-push-pr.yml/badge.svg)](https://github.com/iganets/iganet-perftests/actions/workflows/ci-push-pr.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://iganets.github.io/iganet/)

[![GitHub Releases](https://img.shields.io/github/release/iganets/iganet-perftests.svg)](https://github.com/iganets/iganet-perftests/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/iganets/iganet-perftests/total)](https://github.com/iganets/iganet-perftests/releases)
[![GitHub Issues](https://img.shields.io/github/issues/iganets/iganet-perftests.svg)](https://github.com/iganets/iganet-perftests/issues)

This repository contains performance tests for [IGAnets](https://github.com/iganets/iganet), a novel approach to combine the concept of deep operator learning with the mathematical framework of isogeometric analysis.

## Usage instructions

This repository can be used in two modes:

1. As standalone performance tests by running CMake on _this_ repository without flags

2. As optional module in [iganets::core](https://github.com/iganets/iganet) by running CMake on the [iganets::core](https://github.com/iganets/iganet) repository with the flag
   ```
   -DIGANET_OPTIONAL="perftests"
   ```

In both cases, the tests can be run via
```shell
make test
```
By default, all performance tests are disabled and need to be enabled explicitly.

To obtain a list of available tests run (or another executable in the `perftests` folder)
```shell
./perftests/perftest_bspline_eval --gtest_filter="*" --gtest_list_tests
```

To execute one or more tests run
```shell
./perftests/perftest_bspline_eval --gtest_filter="*UniformBSpline_*parDim1*:-*Non*"
```

This specific command will run all `UniformBSpline` tests with 1 parametric dimension.

The generated performance results can be post-processed with the tools provided in [iganets::perftests_results](https://github.com/iganets/iganet-perftests-results)
