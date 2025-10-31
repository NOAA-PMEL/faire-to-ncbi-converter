# The columns of faire as keys and the columns of ncbi as values (if they map exactly)
ncbi_faire_to_ncbi_column_mappings_exact = {
    "samp_name": "*sample_name",
    "eventDate": "*collection_date",
    "env_broad_scale": "*env_broad_scale",
    "env_local_scale": "*env_local_scale",
    "env_medium": "*env_medium", 
    "geo_loc_name": '*geo_loc_name',
    "collection_method": "samp_collect_method",
    "samp_collect_device": "samp_collect_device",
    "samp_mat_process": "samp_mat_process",
    "samp_store_dur": "samp_store_dur",
    "samp_store_loc": "samp_store_loc",
    "source_material_id": "materialSampleID",
    "tidal_stage": "tidal_stage",
    "neg_cont_type": "neg_cont_type",
    "pos_cont_type": "pos_cont_type",
    "ph": "ph"
}

# For faire_column as keys and ncbit sra columns as values
ncbi_faire_sra_column_mappings_exact = {
    "samp_name": "sample_name",
    "lib_id": "library_ID",
    "filename": "filename",
    "filename2": "filename2"
}

# NCBI sample column name as key and faire columns ask nested values with units
faire_to_ncbi_units = {
    "alkalinity": {
        "faire_col": "tot_alkalinity",
        "faire_unit_col": "tot_alkalinity_unit"
    },
    "ammonium": {
        "faire_col": "ammonium",
        "faire_unit_col": "ammonium_unit"
    },
    "chlorophyll": {
        "faire_col": "chlorophyll",
        "constant_unit_val": "mg/m3"
    },
    "density": {
        "faire_col": "density",
        "faire_unit_col": "density_unit"
    },
    "diss_inorg_carb": {
        "faire_col": "diss_inorg_carb",
        "faire_unit_col": "diss_inorg_carb_unit"
    },
    "diss_inorg_nitro": {
        "faire_col": "diss_inorg_nitro",
        "faire_unit_col": "diss_inorg_nitro_unit"
    },
    "diss_org_carb": {
        "faire_col": "diss_org_carb",
        "faire_unit_col": "diss_org_carb_unit"
    },
    "diss_org_nitro": {
        "faire_col": "diss_org_nitro",
        "faire_unit_col": "diss_org_nitro_unit"
    },
    "diss_oxygen": {
        "faire_col": "diss_oxygen",
        "faire_unit_col": "diss_oxygen_unit"
    },
    "down_par": { # This is a user defined field that we added
        "faire_col": "par",
        "faire_unit_col": "par_unit"
    },
    "elev": {
        "faire_col": "elev",
        "constant_unit_val": "m"
    },
    "light_intensity": {
        "faire_col": "light_intensity",
        "constant_unit_val": "lux"
    },
    "nitrate": {
        "faire_col": "nitrate",
        "faire_unit_col": "nitrate_unit"
    },
    "nitrite": {
        "faire_col": "nitrite",
        "faire_unit_col": "nitrite_unit"
    },
    "nitro": {
        "faire_col": "nitro",
        "faire_unit_col": "nitro_unit"
    },
    "org_carb": {
        "faire_col": "org_carb",
        "faire_unit_col": "org_carb_unit"
    },
    "org_matter": {
        "faire_col": "org_matter",
        "faire_unit_col": "org_matter_unit"
    },
    "org_nitro": {
        "faire_col": "org_nitro",
        "faire_unit_col": "org_nitro_unit"
    },
    "part_org_carb": {
        "faire_col": "part_org_carb",
        "faire_unit_col": "part_org_carb_unit"
    },
    "part_org_nitro": {
        "faire_col": "part_org_nitro",
        "faire_unit_col": "part_org_nitro_unit"
    },
    "phosphate": { #user defined we added
        "faire_col": "phosphate",
        "faire_unit_col": "phosphate_unit"
    },
    "pressure": {
        "faire_col": "pressure",
        "faire_unit_col": "pressure_unit"
    },
    "salinity": {
        "faire_col": "salinity",
        "constant_unit_val": "psu"
    },
    "samp_size": {
        "faire_col": "samp_size",
        "faire_unit_col": "samp_size_unit"
    },
    "samp_store_temp": {
        "faire_col": "samp_store_temp",
        "constant_unit_val": "C"
    },
    "samp_vol_we_dna_ext": {
        "faire_col": "samp_vol_we_dna_ext",
        "faire_unit_col": "samp_vol_we_dna_ext_unit"
    },
    "silicate": { #user defined we added
        "faire_col": "silicate",
        "faire_unit_col": "silicate_unit"
    },
    "size_frac": {
        "faire_col": "size_frac",
        "constant_unit_val": "µm"
    },
    "size_frac_low": {
        "faire_col": "size_frac_low",
        "constant_unit_val": "µm"
    },
    "suspend_part_matter": {
        "faire_col": "suspend_part_matter",
        "constant_unit_val": "mg/L"
    },
    "temp": {
        "faire_col": "temp",
        "constant_unit_val": "C"
    },
    "tot_depth_water_col": {
        "faire_col": "tot_depth_water_col",
        "constant_unit_val": "m"
    },
    "tot_diss_nitro": {
        "faire_col": "tot_diss_nitro",
        "faire_unit_col": "tot_diss_nitro_unit"
    },
    "tot_inorg_nitro": {
        "faire_col": "tot_inorg_nitro",
        "faire_unit_col": "tot_inorg_nitro_unit"
    },
    "tot_nitro": {
        "faire_col": "tot_nitro",
        "faire_unit_col": "tot_nitro_unit"
    },
    "tot_part_carb": {
        "faire_col": "tot_part_carb",
        "faire_unit_col": "tot_part_carb_unit"
    },
    "turbidity": {
        "faire_col": "turbidity",
        "constant_unit_val": "ntu"
    },
    "water_current": {
        "faire_col": "water_current",
        "constant_unit_val": "m/s"
    }
}