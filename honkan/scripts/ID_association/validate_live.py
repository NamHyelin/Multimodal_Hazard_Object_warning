"""Read-only metadata sample; save evidence, no images or Tracker writes."""
import collections,json,time,argparse
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=30);p.add_argument('--output',required=True);args=p.parse_args()
rclpy.init();n=Node('id_association_validation');rows=[]
def cb(m):
    row=json.loads(m.data);row['received_monotonic']=time.monotonic();rows.append(row)
sub=n.create_subscription(String,'/perception/id_association/results',cb,10)
start=time.monotonic()
while time.monotonic()-start<args.seconds:rclpy.spin_once(n,timeout_sec=.1)
summary={}
for cam in ('gate','intersection'):
    a=[r for r in rows if r['camera']==cam];ms=[m for r in a for m in r['matches']]
    summary[cam]=dict(frames=len(a),fps=(len(a)-1)/(a[-1]['received_monotonic']-a[0]['received_monotonic']) if len(a)>1 else 0,
        states=dict(collections.Counter(r['status'] for r in a)),matches=len(ms),confirmed=sum(m['confirmed'] for m in ms),
        unique_uuids=len({m['uuid'] for m in ms}),ambiguous=sum(len(r['ambiguous_yolo']) for r in a),
        mean_processing_ms=sum(r['processing_ms'] for r in a)/max(len(a),1),
        duplicate_assignment_frames=sum(len({m['uuid'] for m in r['matches']})!=len(r['matches']) or len({m['yolo_index'] for m in r['matches']})!=len(r['matches']) for r in a))
out=dict(summary=summary,rows=rows);Path(args.output).write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
n.destroy_node();rclpy.shutdown()
