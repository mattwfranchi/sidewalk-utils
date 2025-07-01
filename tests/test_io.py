"""
Tests for io.py functionality, particularly the Nexar2020Dataset class.
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path

from user.io import Nexar2020Dataset, NEXAR_2020, NEXAR_2023
from utils.logger import get_logger


logger = get_logger(__name__)
logger.setLevel("INFO")

# Define project root for test resources
project_root = Path("/share/ju/sidewalk_utils")


class TestNexar2020Dataset(unittest.TestCase):
    """Test cases for the Nexar2020Dataset class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.dataset = Nexar2020Dataset(load_imgs=False, load_md=False)
        # Check if data paths exist but don't fail test setup if they don't
        self.has_data = os.path.exists(NEXAR_2020)
        if not self.has_data:
            logger.warning(f"Path does not exist: {NEXAR_2020}. Some tests will be skipped.")
        
        # Create temporary test directory
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test dataset initialization with various parameters."""
        logger.info("Testing dataset initialization")
        
        # Test default initialization
        dataset = Nexar2020Dataset()
        self.assertEqual(dataset.ncpus, 8)
        self.assertFalse(dataset.load_imgs)
        self.assertFalse(dataset.load_md)
        self.assertEqual(dataset.imgs, [])
        self.assertEqual(dataset.md, [])
        
        # Test custom initialization
        dataset = Nexar2020Dataset(ncpus=4)
        self.assertEqual(dataset.ncpus, 4)

    def test_load_img_dir(self):
        """Test the load_img_dir method."""
        logger.info("Testing load_img_dir function")
        
        # Create dummy images in the temporary directory
        for i in range(3):
            dummy_img = self.test_dir / f"test{i}.jpg"
            dummy_img.touch()
        
        # Test loading images from directory
        images = self.dataset.load_img_dir(self.test_dir)
        self.assertEqual(len(images), 3)
        
        # Verify we got the correct files
        file_names = [img.name for img in images]
        self.assertIn("test0.jpg", file_names)
        self.assertIn("test1.jpg", file_names)
        self.assertIn("test2.jpg", file_names)

    @unittest.skipIf(not os.path.exists(os.path.join(NEXAR_2020, "imgs/oct_15-nov-15")), 
                     "Nexar 2020 dataset not available for testing")
    def test_load_octnov2020_imgs(self):
        """Test loading images from Nexar 2020 dataset."""
        logger.info("Testing load_octnov2020_imgs function")
        
        # Only test with a small subset by monkeypatching the glob function
        # to limit number of files processed for faster testing
        original_glob = Path.glob
        
        def limited_glob(self, pattern):
            results = list(original_glob(self, pattern))
            if pattern == "*.jpg":  # When looking for images
                return results[:2]  # Only take 2 images per directory
            else:  # When looking for directories
                return results[:5]  # Only take 5 directories
            
        try:
            Path.glob = limited_glob
            
            # Create dataset with loading enabled but limit to 2 CPUs for testing
            dataset = Nexar2020Dataset(load_imgs=True, ncpus=2)
            
            # Verify images were loaded
            self.assertIsNotNone(dataset.imgs)
            self.assertGreater(len(dataset.imgs), 0)
            # Should now have at most 5 directories × 2 images = 10 total images
            self.assertLessEqual(len(dataset.imgs), 10)
            logger.info(f"Loaded {len(dataset.imgs)} images")
            
        finally:
            # Restore original glob function
            Path.glob = original_glob

    @unittest.skipIf(not os.path.exists(os.path.join(NEXAR_2020, "metadata/oct_15-nov-15")), 
                     "Nexar 2020 metadata not available for testing")
    def test_load_octnov2020_md(self):
        """Test loading metadata from Nexar 2020 dataset."""
        logger.info("Testing load_octnov2020_md function")
        
        # Create dataset with metadata loading enabled
        dataset = Nexar2020Dataset(load_md=True, ncpus=2)
        
        # Verify metadata was loaded
        self.assertIsNotNone(dataset.md)
        self.assertGreater(len(dataset.md), 0)
        
        # Check that required columns were created
        self.assertIn('frame_id', dataset.md.columns)
        self.assertIn('timestamp', dataset.md.columns)
        
        logger.info(f"Loaded {len(dataset.md)} metadata rows")

    def test_chunk_list(self):
        """Test the _chunk_list method."""
        logger.info("Testing _chunk_list function")
        
        # Test with list larger than number of chunks
        test_list = list(range(20))
        chunks = self.dataset._chunk_list(test_list, 4)
        
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(chunks[0]), 5)
        self.assertEqual(chunks[0], [0, 1, 2, 3, 4])
        
        # Test with list smaller than number of chunks
        test_list = list(range(3))
        chunks = self.dataset._chunk_list(test_list, 5)
        
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], [0])
        self.assertEqual(chunks[1], [1])
        self.assertEqual(chunks[2], [2])
        
        # Test with empty list
        test_list = []
        chunks = self.dataset._chunk_list(test_list, 3)
        self.assertEqual(chunks, [])

    def test_path_constants(self):
        """Test that the path constants are defined correctly."""
        logger.info("Testing path constants")
        
        self.assertIsInstance(NEXAR_2020, str)
        self.assertIsInstance(NEXAR_2023, str)
        self.assertTrue(len(NEXAR_2020) > 0)
        self.assertTrue(len(NEXAR_2023) > 0)


if __name__ == "__main__":
    # Run tests with more detailed output
    logger.info("Starting IO tests")
    unittest.main(verbosity=2)