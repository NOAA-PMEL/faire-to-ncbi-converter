from ncbi_converter.ncbi_mapper import NCBIMapper

def main() -> None:
    
    ncbi_mapper = NCBIMapper(config_file='/home/poseidon/zalmanek/FAIRe-Mapping/runs/run4/ncbi_mapping/ncbi_config.yaml')
    ncbi_mapper.create_ncbi_submission()

if __name__ == "__main__":
    main()