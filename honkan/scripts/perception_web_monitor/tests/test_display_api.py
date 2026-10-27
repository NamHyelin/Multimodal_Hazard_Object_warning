import json
import sys
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from monitor import Handler,Monitor

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.node=SimpleNamespace(lock=threading.RLock(),lidar_box_mode='2d')
        self.node.display_settings=lambda mode=None:Monitor.display_settings(self.node,mode)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.server.node=self.node
        self.thread=threading.Thread(target=self.server.serve_forever);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)+'/api/display'
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
    def post(self,value):
        request=urllib.request.Request(self.url,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(request,timeout=2))
    def test_default_and_roundtrip(self):
        self.assertEqual(json.load(urllib.request.urlopen(self.url))['lidar_box_mode'],'2d')
        self.assertEqual(self.post({'lidar_box_mode':'3d'})['lidar_box_mode'],'3d')
        self.assertEqual(self.post({'lidar_box_mode':'2d'})['lidar_box_mode'],'2d')
    def test_invalid_rejected_without_change(self):
        for value in ({'lidar_box_mode':'invalid'},{'lidar_box_mode':None},{'lidar_box_mode':[]},{},[]):
            with self.assertRaises(urllib.error.HTTPError) as cm:self.post(value)
            self.assertEqual(cm.exception.code,400)
            self.assertEqual(self.node.lidar_box_mode,'2d')

if __name__=='__main__':unittest.main()
