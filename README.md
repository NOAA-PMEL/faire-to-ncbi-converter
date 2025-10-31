# faire-to-ncbi-converter
Converts metabarcoding FAIRe-formatted sample and experiment run metadata into the required file formats for NCBI BioSample and SRA submissions.

## This tool requires the following documents:
- FAIRe formatted SampleMetadata CSV(s)
- FAIRe formatted ExperimentRunMEtadata CSV(s) that are associated with the samples.
- A Library Preparation BeBOP publicly published in a github repository.
- A filled out config.yaml file to inform the code

## This tool makes these assumptions: 
1. It uses the list of samples in the ExperimentRunMetadata as the source of truth, so if samples exist in the SampleMetadata but not the ExperimentRunMetadata, it will drop thos samples in the final NCBI filled out submission template and list those dropped samples in a folder called 'logging' saved to the directory you are saving the ncbi sample excel file to.
2. The NCBI submission uses one type of library preparation, and that protocol is standardized as a BeBOP and accessible via a public github repository.
3. To create the NCBI sample_title and sra_title it assumes that the FAIRe geo_loc_name, the assay_name, and samp_category are completed. And for negative and positive controls, it assumes that the neg_cont_type, and pos_cont_type are completeted. If they aren't the titles, may look strange or code may fail.

## Future Development
- This tool is really only for metabarcoding, will need to add functionality for qPCR
- This tool has functionlity for OME's OSU runs - will take out after fixing and submitting these as this pipeline is not broadly applicable.

## Installation
- Need to install Conda

1. Clone repo:
```
git clone https://github.com/NOAA-PMEL/faire-to-ncbi-converter.git
cd faire-to-ncbi-converter
```

2. Create Conda Environment
```
conda env create -f environment.yaml
conda activate ncbi-converter
```

3. Edit the config.yaml file

4. Run the code
```
python main.py 
```




