import unittest, pandas as pd, joblib
import model_core as core
class ModelTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.df=pd.read_csv(core.DATA_PATH); cls.bundle=joblib.load(core.MODEL_PATH)
 def test_dataset_rows(self): self.assertEqual(len(self.df),836)
 def test_brand_model_columns(self): self.assertIn('brand',self.df); self.assertIn('model',self.df)
 def test_prediction_positive(self):
  row=self.df.iloc[[0]][core.FEATURES]; self.assertGreater(float(self.bundle['model'].predict(row)[0]),0)
 def test_model_metrics_exist(self): self.assertIn(self.bundle['best_model'],core.MODEL_NAMES)
if __name__=='__main__':unittest.main()
