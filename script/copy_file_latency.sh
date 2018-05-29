#!/bin/bash

FILE=data/small_text.txt
STAGING_DIR=/mnt/pmem0/staging/

rm -f ${STAGING_DIR}/*

for i in {1..10}; do
  echo "cp ${FILE} ${STAGING_DIR}/90m_text${i}.txt"
  TIMESTAMP=$(date +%s%N | cut -b1-16)
  echo "  ${TIMESTAMP}"
  cp ${FILE} ${STAGING_DIR}/small_text${i}.txt
  sleep 1
done