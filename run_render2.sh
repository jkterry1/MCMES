mkdir -p ./mature_simulations
mkdir -p ./render_logs
rm -rf mature_simulations/*
rm -rf render_logs/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 render.py 4 &> ./render_logs/render_4.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 render.py 5 &> ./render_logs/render_5.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 render.py 6 &> ./render_logs/render_6.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 render.py 7 &> ./render_logs/render_7.out &
