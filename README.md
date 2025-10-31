# faire-to-ncbi-converter
Converts FAIRe-formatted sample and experiment run metadata into the required file formats for NCBI BioSample and SRA submissions.

This tool requires the following documents:
- FAIRe formatted SampleMetadata CSV(s)
- FAIRe formatted ExperimentRunMEtadata CSV(s) that are associated with the samples.
- A Library Preparation BeBOP publicly published in a github repository.
- A filled out config.yaml file to inform the code

This tool makes these assumptions: 
1. It uses the list of samples in the ExperimentRunMetadata as the source of truth, so if samples exist in the SampleMetadata but not the ExperimentRunMetadata, it will drop thos samples in the final NCBI filled out submission template and list those dropped samples in a folder called 'logging' saved to the directory you are saving the ncbi sample excel file to.
2. The NCBI submission uses one type of library preparation, and that protocol is standardized as a BeBOP and accessible via a public github repository.


