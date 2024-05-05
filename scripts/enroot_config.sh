#!/usr/bin/env bash

environ() {
    # Keep all the environment from the host
    # env
    comm -3 <(declare | sort) <(declare -f | sort)
}
