mkdir -p ./optimization_gifs
rm -rf optimization_gifs/*

OMP_NUM_THREADS=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python3 render_optimization_policies.py &> render_optimization_log.out &
