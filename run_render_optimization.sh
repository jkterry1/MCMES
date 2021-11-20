mkdir -p ./optimization_gifs
rm -rf optimization_gifs/*

python3 nohup render_optimization_policies.py &> render_optimization_log.out &