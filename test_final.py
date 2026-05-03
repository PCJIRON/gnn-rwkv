import asyncio
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Set UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from chat import crawl_and_train

async def test_run():
    mock_brain = MagicMock()
    print("Testing /train with 'ben 10'...")
    
    # Mock input to return 1 (epoch) automatically
    with patch('builtins.input', return_value='1'):
        try:
            await crawl_and_train("ben 10", mock_brain)
            print("\nTEST SUCCESSFUL: Model learned from Ben 10 data!")
        except Exception as e:
            print(f"\nTEST FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_run())
