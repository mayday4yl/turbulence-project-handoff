"""Small tests of the new scientific data contract before remote generation."""
import io
import unittest
import numpy as np
from PIL import Image
from build_dataset import preprocess, xi_seed, source_order, subset_order


def png(array):
    stream=io.BytesIO()
    Image.fromarray(array).save(stream,format='PNG')
    return stream.getvalue()


class ContractTests(unittest.TestCase):
    def test_identity_preprocessing(self):
        a=np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3)
        x,m=preprocess(png(a))
        self.assertTrue(np.array_equal(x,a.transpose(2,0,1).astype(np.float32)/np.float32(255)))
        self.assertFalse(m['resize_applied'])
        self.assertFalse(m['crop_applied'])

    def test_odd_aspect_floor_center(self):
        x,m=preprocess(png(np.zeros((301,499,3),dtype=np.uint8)))
        self.assertEqual(m['resize_size_wh'],[424,256])
        self.assertEqual(m['crop_box_xyxy'],[84,0,340,256])
        self.assertEqual(x.shape,(3,256,256))

    def test_portrait_and_small_input(self):
        x,m=preprocess(png(np.zeros((105,63,3),dtype=np.uint8)))
        self.assertEqual(m['resize_size_wh'],[256,426])
        self.assertEqual(m['crop_box_xyxy'],[0,85,256,341])

    def test_seed_mapping_and_order(self):
        values=[xi_seed(f'source_{s}.jpg',r,b) for s in range(100) for r in range(1,5) for b in ('epsilon_zernike','epsilon_tilt')]
        self.assertEqual(len(set(values)),800)
        self.assertTrue(all(0<=n<(1<<63) for n in values))
        self.assertEqual(xi_seed('test.jpg',1,'epsilon_tilt'),xi_seed('test.jpg',1,'epsilon_tilt'))
        ids=[f'item{i}' for i in range(100)]
        self.assertEqual(sorted(ids,key=source_order),sorted(reversed(ids),key=source_order))
        self.assertEqual(sorted(ids,key=subset_order),sorted(reversed(ids),key=subset_order))


if __name__=='__main__':
    unittest.main()
