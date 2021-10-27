mkdir -p ./single_run_logs
rm -rf single_run_logs/*

OMP_NUM_THREADS=1 nohup python3 single_run.py &> ./single_run_logs/log.out &

