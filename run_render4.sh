mkdir -p ./mature_simulations
mkdir -p ./render_logs
rm -rf mature_simulations/*
rm -rf render_logs/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 render.py 12 &> ./render_logs/render_12.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 render.py 13 &> ./render_logs/render_13.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 render.py 14 &> ./render_logs/render_14.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 render.py 15 &> ./render_logs/render_15.out &
