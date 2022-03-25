mkdir -p ./results
rm -rf results/*

OMP_NUM_THREADS=1 nohup python3 render_optimization_policies.py &> ./render_optimization_logs.out &
