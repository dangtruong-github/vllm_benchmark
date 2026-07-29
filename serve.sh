#!/bin/bash

vllm serve Qwen/Qwen3.5-0.8B \
    --max-model-len=32678 \
    --gpu-memory-utilization=0.95 \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=8192 \
    --enable-prefix-caching \
    --scheduling-policy=priority \
    --enable-chunked-prefill \
    --max-num-seqs=8 \
    --kv-cache-dtype=fp8 > vllm.log
