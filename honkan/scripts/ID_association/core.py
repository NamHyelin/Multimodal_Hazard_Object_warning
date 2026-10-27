"""CPU geometry matching, independent of ROS. IDs are always supplied by Tracker."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    wh=np.maximum(0.,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter=float(np.prod(wh))
    area=lambda r:float(np.prod(np.maximum(0.,r[2:]-r[:2])))
    return inter/max(area(a)+area(b)-inter,1e-9)


def associate(tracks,detections,previous=None,ambiguity=.08):
    """Class + overlap/center gates, Hungarian assignment, conservative ambiguity rejection."""
    if not tracks or not detections:return [],[]
    previous=previous or {}; cost=np.full((len(tracks),len(detections)),1e6)
    for i,tr in enumerate(tracks):
        for j,det in enumerate(detections):
            acls,bcls=tr['label'],det['label']
            if acls!=bcls and not (acls in (1,2,3,4) and bcls in (1,2,3,4)):continue
            a,b=np.asarray(tr['box']),np.asarray(det['box']); overlap=iou(a,b)
            distance=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2))
            norm=distance/max(float(np.linalg.norm(b[2:]-b[:2])),15.)
            if norm>1 or (overlap<.01 and norm>.5):continue
            value=.65*(1-overlap)+.35*min(norm,1.)
            old=previous.get(tr['uuid'])
            if old is not None and iou(old,b)>.2:value-=.04
            if value<.85:cost[i,j]=value
    # Unmatched dummy columns keep invalid pairs from stealing another target's detection.
    rows,cols=linear_sum_assignment(np.concatenate([cost,np.full((len(tracks),len(tracks)),.85)],axis=1))
    matches,ambiguous=[],[]
    for i,j in zip(rows,cols):
        if j>=len(detections) or cost[i,j]>=.85:continue
        alternatives=np.r_[np.delete(cost[i],j),np.delete(cost[:,j],i)]
        if alternatives.size and np.min(alternatives)-cost[i,j]<ambiguity:
            ambiguous.append(detections[j]['index']);continue
        matches.append((int(i),int(j),float(cost[i,j])))
    return matches,ambiguous


def rotation(q):
    v=np.array([q.x,q.y,q.z,q.w],dtype=float); norm=np.linalg.norm(v)
    if norm<1e-9:return np.eye(3)
    x,y,z,w=v/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


class Confirmation:
    def __init__(self,required=3):self.required=required;self.memory={}

    def update(self,matches,t):
        active={}
        for uuid,box in matches:
            old=self.memory.get(uuid)
            continuous=old is not None and 0<t-old['t']<=.3 and iou(old['box'],box)>.1
            hits=old['hits']+1 if continuous else 1
            active[uuid]=dict(t=t,box=box,hits=hits,confirmed=hits>=self.required)
        # No held display IDs: absence/ambiguity resets the evidence on the next processed frame.
        self.memory=active
        return active
