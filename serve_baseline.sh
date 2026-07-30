#!/bin/bash

vllm serve Qwen/Qwen3.5-2B \
    --max-model-len=30000 \
    --gpu-memory-utilization=0.9 \
    --tensor-parallel-size=1 \
    --enable-prefix-caching \
    --scheduling-policy=priority \
    --enable-chunked-prefill \
    --max-num-seqs=1 > vllm.log
