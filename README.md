# MoMA preprocessing

This project contains the code to preprocess OME-TIFF image stacks from Mother machine experiments before analysis with [MoMA (Mother Machine Analyzer)](https://github.com/nimwegenLab/moma).

## Table of Contents

- [MoMA preprocessing](#moma-preprocessing)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Usage](#usage)

## About

[MoMA (Mother Machine Analyzer)](https://github.com/nimwegenLab/moma) loads and processes each growthlane of a Mother machine experiment separately.
The software in this repository preprocesses the full-frame OME-TIFF stacks, which are recorded using MicroManager during Mother machine experiments, and splits them into separate growthlanes.

## Getting Started

## Installation

MoMA preprocessing is available as Docker image. It is recommended that you use the Docker image for quick and easy setup as described [here](https://github.com/nimwegenLab/moma-preprocessing-module).

Alternatively, you can look at the [Dockerfile](https://github.com/nimwegenLab/moma-preprocessing/blob/master/Dockerfile) in this repository, if you want to understand how to set up MoMA preprocessing without container.

## Usage

Usage information is available [here](https://github.com/nimwegenLab/moma/wiki/preprocessing) and in the [MoMA tutorial](https://github.com/nimwegenLab/moma/wiki/MoMA-introductory-tutorial).

