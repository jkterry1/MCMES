mkdir -p ./mature_simulations
mkdir -p ./render_logs
rm -rf mature_simulations/*
rm -rf render_logs/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 render.py 0 &> ./render_logs/render_0.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 render.py 1 &> ./render_logs/render_1.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 render.py 2 &> ./render_logs/render_2.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 render.py 3 &> ./render_logs/render_3.out &