@echo off
echo test

python main.py --n_runs 10 --dataset qm8 --split scaffold
python main.py --n_runs 10 --dataset qm8