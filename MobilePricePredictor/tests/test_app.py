import unittest
import app as web
class AppTests(unittest.TestCase):
 def setUp(self): self.c=web.app.test_client()
 def test_home(self): self.assertEqual(self.c.get('/').status_code,200)
 def test_schema(self): self.assertIn('brands',self.c.get('/api/schema').json)
 def test_analytics(self): self.assertGreater(self.c.get('/api/analytics').json['records'],0)
 def test_phone_lookup(self):
  s=self.c.get('/api/schema').json; b=s['brands'][0]; m=self.c.get('/api/models/'+b).json['models'][0]; self.assertEqual(self.c.get('/api/phone?brand='+b+'&model='+m).status_code,200)
if __name__=='__main__':unittest.main()
