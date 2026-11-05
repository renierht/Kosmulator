import unittest
from Plots.Plot_functions import validated_rd_plot_band
class PlotBandTests(unittest.TestCase):
 def test_valid(self):
  self.assertEqual(validated_rd_plot_band(1,2,3,lambda p:140.),(2,3,None))
 def test_bad_endpoint_omits_not_substitutes(self):
  def resolver(p):
   if p==2:raise ValueError('invalid CLASS branch')
   return 140.
  a,b,note=validated_rd_plot_band(1,2,3,resolver)
  self.assertIsNone(a);self.assertIsNone(b);self.assertIn('omitted',note)
 def test_bad_median_propagates(self):
  with self.assertRaises(ValueError):validated_rd_plot_band(1,2,3,lambda p:float('nan'))
 def test_programming_error_not_hidden(self):
  def resolver(p):
   if p==2:raise TypeError('bug')
   return 140.
  with self.assertRaises(TypeError):validated_rd_plot_band(1,2,3,resolver)
