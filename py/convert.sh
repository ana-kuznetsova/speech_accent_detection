#!/bin/bash

for f in *
    do shortname=`echo $f | cut -d '.' -f 1`
        ffmpeg -i $f $shortname.wav
    done