cd ~/hyelin
source .venv/bin/activate

# operate dt_server.py first, and then operate honkan server code

# Terminal 1
python receiver.py   --host 0.0.0.0   --port 8001   --exchange_dir /home/morita/hyelin/dt_exchange

# Terminal 2
python dt_server.py