import unittest
from core import associate,Confirmation,iou

def tr(u,b,c=7):return dict(uuid=u,box=b,label=c)
def det(i,b,c=7):return dict(index=i,box=b,label=c)

class Tests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(iou([0,0,0,0],[0,0,0,0]),0)
    def test_empty(self):self.assertEqual(associate([],[]),([],[]))
    def test_unique(self):
        m,a=associate([tr('a',[0,0,20,40]),tr('b',[100,0,120,40])],
                      [det(0,[100,0,120,40]),det(1,[0,0,20,40])])
        self.assertEqual([(i,j) for i,j,_ in m],[(0,1),(1,0)])
        self.assertEqual(a,[])
    def test_wrong_class_and_far(self):
        self.assertEqual(associate([tr('a',[0,0,20,40])],[det(0,[0,0,20,40],1)])[0],[])
        self.assertEqual(associate([tr('a',[0,0,20,40])],[det(0,[500,0,520,40])])[0],[])
    def test_ambiguous(self):
        m,a=associate([tr('a',[0,0,20,40]),tr('b',[0,0,20,40])],[det(0,[0,0,20,40])])
        self.assertEqual(m,[]);self.assertEqual(a,[0])
    def test_one_to_one(self):
        m,_=associate([tr('a',[0,0,20,40])],[det(0,[0,0,20,40]),det(1,[0,0,20,40])])
        self.assertEqual(m,[])
    def test_confirmation_reset(self):
        c=Confirmation();b=[0,0,20,40]
        self.assertFalse(c.update([('a',b)],1)['a']['confirmed'])
        self.assertFalse(c.update([('a',b)],1.1)['a']['confirmed'])
        self.assertTrue(c.update([('a',b)],1.2)['a']['confirmed'])
        c.update([],1.3)
        self.assertEqual(c.update([('a',b)],1.4)['a']['hits'],1)
    def test_gap_and_duplicate(self):
        c=Confirmation();b=[0,0,20,40]
        c.update([('a',b)],1)
        self.assertEqual(c.update([('a',b)],1)['a']['hits'],1)
        self.assertEqual(c.update([('a',b)],2)['a']['hits'],1)

if __name__=='__main__':unittest.main()
