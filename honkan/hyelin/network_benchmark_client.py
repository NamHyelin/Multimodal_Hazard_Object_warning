"""Paired text/JPEG transport microbenchmark, separate from live UDP pipeline."""
import csv, json, random, socket, struct, time
from pathlib import Path

ROOT = Path('/autoware_runtime/hyelin/results/centerpoint_florence/calibration_pilot_20261022_01')
def read_exact(conn, size):
    data = bytearray()
    while len(data)<size:
        part=conn.recv(size-len(data))
        if not part: raise EOFError
        data.extend(part)
    return bytes(data)

rows = []
with socket.create_connection(('172.16.1.203',8002), timeout=10) as s:
    s.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    def measure(body):
        t1=time.time(); m1=time.perf_counter()
        s.sendall(struct.pack('!I',len(body))+body)
        t2,t3=struct.unpack('!dd', read_exact(s,16))
        m4=time.perf_counter(); t4=time.time()
        return dict(t1=t1,t2=t2,t3=t3,t4=t4,rtt_ms=(m4-m1)*1000,
            offset_lower=t3-t4,offset_upper=t2-t1)
    for i in range(40):
        rows.append(dict(kind='clock_probe',frame=-1,obj=-1,rep=i,bytes=1,**measure(b'x')))
        time.sleep(.01)
    rng = random.Random(42)
    frames=[json.loads(line) for line in (ROOT/'frames.jsonl').read_text().splitlines()]
    for frame in frames:
        for obj in frame['objects']:
            if not obj.get('florence_triggered'): continue
            crop=list((ROOT/'florence_crops').glob(f"frame_{frame['frame_index']:06d}_*object_{obj['object_index']:02d}_*.jpg"))
            if len(crop)!=1: continue
            metadata=json.dumps(dict(frame_index=frame['frame_index'],object_index=obj['object_index'],
                world_position_m=obj['world_position_xyz'],size_m=obj['dimensions_xyz']),separators=(',',':')).encode()
            common=struct.pack('!I',len(metadata))+metadata
            bodies={'text':common+obj['florence_caption'].encode(), 'image':common+crop[0].read_bytes()}
            for rep in range(3):
                order=['text','image']; rng.shuffle(order)
                for kind in order:
                    rows.append(dict(kind=kind,frame=frame['frame_index'],obj=obj['object_index'],
                        rep=rep,bytes=len(bodies[kind]),**measure(bodies[kind])))
                    time.sleep(.01)
    for i in range(40):
        rows.append(dict(kind='clock_probe_end',frame=-1,obj=-1,rep=i,bytes=1,**measure(b'x')))
        time.sleep(.01)
    s.sendall(struct.pack('!I',0))
with (ROOT/'network_benchmark.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(f'Saved {len(rows)} benchmark rows',flush=True)
