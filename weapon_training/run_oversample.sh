#!/bin/bash
exec 9>/tmp/oversample.lock
flock -n 9 || { echo "jiny beh drzi zamek, koncim"; exit 1; }
~/yolo-train-venv/bin/python ~/weapon_training/oversample_knife.py
echo "EXIT=$?"
