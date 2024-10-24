import unittest

from assertpy import assert_that
from polnet.stomo import MmerFile, MmerMbFile

class TestMmerFile(unittest.TestCase):

    def test_monomer_attributes_are_set_correctly(self):
        monomer_in_test = MmerFile('data/in_10A/1bxn_10A.pns')
        assert_that(monomer_in_test.get_iso()).is_equal_to(0.1)
        assert_that(monomer_in_test.get_mmer_id()).is_equal_to('pdb_1bxn')
        assert_that(monomer_in_test.get_mmer_svol()).\
            is_equal_to('/templates/mrcs_10A/1bxn.mrc')
        assert_that(monomer_in_test.get_pmer_occ()).is_equal_to(0.5)
    
    def test_invalid_monomer_file_extension(self):
        with self.assertRaises(AssertionError):
            MmerFile('invalid.txt')

class TestMmerMbFile(unittest.TestCase):

    def test_mb_protein_attributes_are_set_correctly(self):
        membrane_protein_in_test = MmerMbFile('data/in_10A/mb_4pe5_10A.pms')
        assert_that(membrane_protein_in_test.get_mmer_id()).is_equal_to('pdb_mb_4pe5')
        assert_that(membrane_protein_in_test.get_pmer_reverse_normals()).is_equal_to(True)

    def test_invalid_mb_protein_file_extension(self):
        with self.assertRaises(AssertionError):
            MmerMbFile('invalid.txt')

if __name__ == '__main__':
    unittest.main()
