# Strided.jl

[![Strided](http://pkg.julialang.org/badges/Strided_0.7.svg)](http://pkg.julialang.org/?pkg=Strided)
[![Build Status](https://travis-ci.org/Jutho/Strided.jl.svg?branch=master)](https://travis-ci.org/Jutho/Strided.jl)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Coverage Status](https://coveralls.io/repos/Jutho/Strided.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/Jutho/Strided.jl?branch=master)
[![codecov.io](http://codecov.io/github/Jutho/Strided.jl/coverage.svg?branch=master)](http://codecov.io/github/Jutho/Strided.jl?branch=master)

A Julia package for working more efficiently with strided arrays, i.e. dense arrays
whose memory layout has a fixed stride along every dimension. Strided.jl does not
make any assumptions about the strides (such as stride 1 along first dimension, or
monotoneously increasing strides) and provides multithreaded and cache friendly
implementations for mapping, reducing, broadcasting such arrays, as well as taking
views and reshaping and permuting dimensions.
