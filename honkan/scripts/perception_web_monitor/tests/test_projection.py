import sys
import unittest
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from projection import box_geometry

K=np.array([[100.,0,100],[0,100,100],[0,0,1]])
SIGNS=np.array([[1 if i&b else -1 for b in (1,2,4)] for i in range(8)])

class GeometryTests(unittest.TestCase):
    def test_envelope_contains_all_corners(self):
        c=SIGNS.astype(float)+[0,0,8]
        segments,box=box_geometry(c,K,np.zeros(5),200,200)
        uv=cv2.projectPoints(c,np.zeros(3),np.zeros(3),K,np.zeros(5))[0].reshape(-1,2)
        self.assertEqual(len(segments),12)
        np.testing.assert_array_equal(box[0],np.floor(uv.min(axis=0)))
        np.testing.assert_array_equal(box[1],np.ceil(uv.max(axis=0)))
    def test_off_screen(self):
        self.assertEqual(box_geometry(SIGNS.astype(float)+[100,0,8],K,np.zeros(5),200,200),([],None))
    def test_behind(self):
        self.assertEqual(box_geometry(SIGNS.astype(float)+[0,0,-8],K,np.zeros(5),200,200),([],None))
    def test_partial_image_clip(self):
        _,box=box_geometry(SIGNS.astype(float)+[-7,0,8],K,np.zeros(5),200,200)
        self.assertEqual(box[0][0],0)
        self.assertLess(box[1][0],200)
    def test_near_plane(self):
        segments,box=box_geometry(SIGNS.astype(float)+[0,0,.5],K,np.zeros(5),200,200)
        self.assertEqual(box,((0,0),(199,199)))
    def test_nan(self):
        c=SIGNS.astype(float);c[0,0]=float('nan')
        self.assertEqual(box_geometry(c,K,np.zeros(5),200,200),([],None))

if __name__=='__main__':unittest.main()
