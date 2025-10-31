import argparse
import pandas as pd
import sys
sys.path.append("../../..")
from utils.ncbi_mapper import NCBIMapper

def main() -> None:
    
    ncbi_mapper = NCBIMapper(config_file='config.yaml')
    ncbi_mapper.create_ncbi_submission()

if __name__ == "__main__":
    main()