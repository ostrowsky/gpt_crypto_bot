import unittest
from prepare_mission_market import verify_grid

class GridTests(unittest.TestCase):
    def test_gaps_duplicates_and_invalid_prices_are_not_complete(self):
        row=dict(t=0,o=1.,h=2.,l=.5,c=1.,v=2.)
        verify_grid([row],900000,0,900000)
        for rows in ([],[row,row],[dict(row,c=3.)],[dict(row,o=float('nan'))]):
            with self.assertRaises(ValueError):verify_grid(rows,900000,0,900000)

if __name__=='__main__':unittest.main()
