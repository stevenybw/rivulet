#!/bin/bash

FILE=data/90m_text.txt
STAGING_DIR=/mnt/pmem0/staging/

rm -f ${STAGING_DIR}/*

for i in {1..50}; do
  echo "cp ${FILE} ${STAGING_DIR}/90m_text${i}.txt"
  cp ${FILE} ${STAGING_DIR}/90m_text${i}.txt
done
