#!/usr/bin/env python3
"""Read-only Tracker UUID -> YOLO association. Subscribes to metadata, never images."""
import collections
import json
import time
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as Q
from rclpy.time import Time
from tf2_ros import Buffer,TransformListener
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from autoware_perception_msgs.msg import TrackedObjects
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from core import associate,rotation,Confirmation

CFG=json.loads(Path(__file__).with_name('config.json').read_text())
LABELS=['UNKNOWN','CAR','TRUCK','BUS','TRAILER','MOTORCYCLE','BICYCLE','PEDESTRIAN']
def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
def category(o):return max(o.classification,key=lambda c:c.probability).label if o.classification else 0


class Association(Node):
    def __init__(self):
        super().__init__('id_association_v1')
        self.tracks=collections.deque(maxlen=40)
        self.pending={c:collections.deque(maxlen=5) for c in ('gate','intersection')}
        self.info={};self.history={c:Confirmation(CFG['confirmation_frames']) for c in self.pending}
        self.display_ids=collections.OrderedDict();self.next_id=1
        self.last_stamp={c:None for c in self.pending}
        self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.pub=self.create_publisher(String,CFG['output_topic'],10)
        self.create_subscription(TrackedObjects,CFG['track_topic'],self.tracks.append,Q)
        for c in self.pending:
            self.create_subscription(CameraInfo,f'/rsu1/{c}/camera_info',lambda m,c=c:self.info.update({c:m}),Q)
            self.create_subscription(DetectedObjectsWithFeature,f'/rsu1/{c}/yolo26s/objects',lambda m,c=c:self.pending[c].append((time.monotonic(),m)),Q)
        self.create_timer(.01,self.tick)
        self.get_logger().info('ID association V1: geometry only; no velocity extrapolation or Tracker feedback')

    def project(self,msg,info):
        tf=self.tf.lookup_transform(info.header.frame_id,msg.header.frame_id,Time.from_msg(msg.header.stamp)).transform
        rot=rotation(tf.rotation);trans=np.array([tf.translation.x,tf.translation.y,tf.translation.z])
        k=np.array(info.k).reshape(3,3);d=np.array(info.d)
        signs=np.array([[1 if i&bit else -1 for bit in (1,2,4)] for i in range(8)])
        out=[]
        for o in msg.objects:
            if o.existence_probability<CFG['min_track_score']:continue
            p=o.kinematics.pose_with_covariance.pose;dim=o.shape.dimensions
            dims=np.array([dim.x,dim.y,dim.z]);center=np.array([p.position.x,p.position.y,p.position.z])
            if not np.isfinite(dims).all() or np.any(dims<=0) or not np.isfinite(center).all():continue
            r=rotation(p.orientation) if o.kinematics.orientation_availability else np.eye(3)
            corners=(signs*dims/2@r.T+center)@rot.T+trans
            if not np.isfinite(corners).all() or np.any(corners[:,2]<.5):continue
            # Avoid distortion polynomial extrapolation far beyond the calibrated FOV.
            normalized=corners[:,:2]/corners[:,2:]
            radius=((info.k[2]/info.k[0])**2+(info.k[5]/info.k[4])**2)*1.44
            if np.any(np.sum(normalized**2,axis=1)>radius):continue
            uv=cv2.projectPoints(corners,np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2)
            if not np.isfinite(uv).all():continue
            box=np.clip(np.r_[uv.min(axis=0),uv.max(axis=0)],[0,0,0,0],[info.width,info.height,info.width,info.height])
            area=(box[2]-box[0])*(box[3]-box[1])
            if area<16 or area>info.width*info.height*.55:continue
            uuid=bytes(o.object_id.uuid).hex()
            if uuid not in self.display_ids:
                self.display_ids[uuid]=str(self.next_id);self.next_id+=1
                if len(self.display_ids)>10000:self.display_ids.popitem(last=False)
            self.display_ids.move_to_end(uuid)
            out.append(dict(uuid=uuid,display_id=self.display_ids[uuid],label=category(o),box=box.tolist()))
        return out

    def process(self,camera,yolo):
        begin=time.monotonic();t=stamp(yolo)
        result=dict(camera=camera,stamp_sec=yolo.header.stamp.sec,stamp_nanosec=yolo.header.stamp.nanosec,image_stamp=t,
                    status='OK',matches=[],unmatched_yolo=[],ambiguous_yolo=[],projected_tracks=[])
        if self.last_stamp[camera] is not None and t<=self.last_stamp[camera]:return
        self.last_stamp[camera]=t;info=self.info.get(camera)
        if self.get_clock().now().nanoseconds/1e9-t>CFG['max_input_age_sec']:result['status']='STALE_INPUT'
        elif info is None or info.header.frame_id!=yolo.header.frame_id or info.distortion_model not in ('plumb_bob','rational_polynomial') or info.k[0]<=0:
            result['status']='CALIBRATION_MISSING_OR_MISMATCH'
        elif not self.tracks:result['status']='NO_TRACKS'
        else:
            track=min(self.tracks,key=lambda m:abs(stamp(m)-t));dt=stamp(track)-t
            result['track_dt_ms']=round(dt*1000,2)
            if abs(dt)>CFG['track_tolerance_sec']:result['status']='UNSYNC'
            else:
                try:
                    projected=self.project(track,info);detections=[]
                    for i,item in enumerate(yolo.feature_objects):
                        roi=item.feature.roi;obj=item.object
                        if obj.existence_probability<CFG['min_yolo_score'] or roi.width==0 or roi.height==0:continue
                        detections.append(dict(index=i,label=category(obj),box=[roi.x_offset,roi.y_offset,roi.x_offset+roi.width,roi.y_offset+roi.height]))
                    previous={u:v['box'] for u,v in self.history[camera].memory.items() if 0<t-v['t']<=.3}
                    matched,ambiguous=associate(projected,detections,previous,CFG['ambiguity_margin'])
                    memory=self.history[camera].update([(projected[i]['uuid'],detections[j]['box']) for i,j,_ in matched],t)
                    for i,j,cost in matched:
                        tr=projected[i];det=detections[j];state=memory[tr['uuid']]
                        result['matches'].append(dict(uuid=tr['uuid'],display_id=tr['display_id'],yolo_index=det['index'],
                            label=LABELS[det['label']] if det['label']<len(LABELS) else 'UNKNOWN',yolo_box=det['box'],
                            projected_box=tr['box'],cost=round(cost,4),confirmed=state['confirmed'],hits=state['hits']))
                    used={m['yolo_index'] for m in result['matches']}
                    result.update(projected_tracks=projected,ambiguous_yolo=ambiguous,unmatched_yolo=[d['index'] for d in detections if d['index'] not in used])
                except Exception as exc:result['status']='PROJECTION_ERROR';result['error']=str(exc)
        if result['status']!='OK':self.history[camera].update([],t)
        result['processing_ms']=round((time.monotonic()-begin)*1000,2)
        result['result_age_ms']=round((self.get_clock().now().nanoseconds/1e9-t)*1000,2)
        self.pub.publish(String(data=json.dumps(result,allow_nan=False)))

    def tick(self):
        for camera,q in self.pending.items():
            if q and time.monotonic()-q[0][0]>=CFG['result_wait_sec']:
                _,msg=q.popleft();self.process(camera,msg)


def main():
    rclpy.init();node=Association()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
