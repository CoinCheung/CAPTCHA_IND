#!/bin/bash
# filename: clear.sh

if [ "$1" == "model" ];then
    rm model_export/*
fi

if [ "$1" == "data" ];then
    rm datasets/*rec datasets/*idx datasets/*lst
fi
