#!/bin/bash

rm -rf hls_proj vivado_hls.log

vivado_hls -f acc_hls.tcl
