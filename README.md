# IgANets-PerfTests

[![GitlabSync](https://github.com/IgANets/iganet-perftests/actions/workflows/gitlab-sync.yml/badge.svg)](https://github.com/IgANets/iganet-perftests/actions/workflows/gitlab-sync.yml)
[![CMake on multiple platforms](https://github.com/IgANets/iganet-perftests/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/IgANets/iganet-perftests/actions/workflows/cmake-multi-platform.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://iganets.github.io/iganet/)

[![GitHub Releases](https://img.shields.io/github/release/iganets/iganet-perftests.svg)](https://github.com/iganets/iganet-perftests/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/iganets/iganet-perftests/total)](https://github.com/iganets/iganet-perftests/releases)
[![GitHub Issues](https://img.shields.io/github/issues/iganets/iganet-perftests.svg)](https://github.com/iganets/iganet-perftests/issues)

This repository contains performance tests for [IgANets](https://github.com/iganets/iganet), a novel approach to combine the concept of deep operator learning with the mathematical framework of isogeometric analysis.

## Usage instructions

This repository can be used in two modes:

1. As standalone performance tests by running CMake on _this_ repository without flags

2. As optional module in [IgANets::core](https://github.com/iganets/iganet) by running CMake on the [IgANets::core](https://github.com/iganets/iganet) repository with the flag
   ```
   -DIGANET_OPTIONAL="perftests"
   ```

In both cases, the tests can be run via
```shell
make test
```

The generated performance results can be post-processed with the tools provided in [IgANets::perftests_results](https://github.com/iganets/iganet-perftests-results)