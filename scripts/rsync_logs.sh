#!/usr/bin/env bash

# Example of usage: nohup ./rsync_logs.sh >> ~/rsync_logs.log &

if [ "$(whoami)" == "akaminski" ]; then
    SOURCE=~/../jfoltyn/Universal_AVR_Model/logs/
    TARGET=~/../akaminski/Universal_AVR_Model/logs/
else
    SOURCE=~/../akaminski/Universal_AVR_Model/logs/
    TARGET=~/../jfoltyn/Universal_AVR_Model/logs/
fi

while true; do
    echo "Syncing logs..."
    echo $(date "+%Y-%m-%d %H:%M:%S")
    rsync --append --groupmap=*:mandziuk-lab --chmod=g=r -rghtv ${SOURCE} ${TARGET}
    echo "Logs synced."
    echo $(date "+%Y-%m-%d %H:%M:%S")
    echo "--------------------------------"
    sleep 60
done
