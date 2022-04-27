#!/bin/bash
curl https://data.ntpc.gov.tw/api/datasets/E09B35A5-A738-48CC-B0F5-570B67AD9C78/csv/file -o ~/ntpc_parking/realtime/parking_lot/$(date +%Y%m%d_%H%M).csv
# curl https://data.ntpc.gov.tw/api/datasets/54A507C4-C038-41B5-BF60-BBECB9D052C6/csv/file -o ~/ntpc_parking/realtime/on_street/$(date +%Y%m%d_%H%M).csv
