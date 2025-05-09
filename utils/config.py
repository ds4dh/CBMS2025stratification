import json 
import re 
import math
from tqdm import tqdm 
import sys, os, logging
import pandas as pd 

import time

# Step 3: Plot the Updated Data
# Define color and shape maps
final_color_map = {
    "eicu": "#C71585",  # Darker Pink
    "miiv": "#0059b3",
    "sic": "#006400"    # Mild Green
}

shape_map = {
    "medoid": "s",    # Square
    "centroid": "^",  # Triangle
    "majority_vote": "o"   # Circle
}
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Starting function: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Finished function: {func.__name__} in {end_time - start_time:.2f}s")
        return result
    return wrapper

if sys.platform == 'darwin':   # is os is darwing (mac) then
    DATASETS = ('mimic_demo','eicu_demo')  # Add more datasets as needed, e.g., ('mimic_iv', 'sic', 'hirid', 'eicu')
else: # in sys.platform linux all datasets  
    DATASETS = ( 'miiv', 'sic', 'eicu', 'hirid')  # Add more datasets as needed, e.g., ('mimic_iv', 'sic', 'hirid', 'eicu')
print('config imported ')
DATASETS = ('mimic_demo','eicu_demo')  # Add more datasets as needed, e.g., ('mimic_iv', 'sic', 'hirid', 'eicu')
DEFAULT_DATASETS = DATASETS

dict_pivot = {'eicu':2, 'hirid':3, 'miiv':4, 'mimic_demo':5, 'sic':6, 'eicu_demo':7, 'mimic':8}

logging.basicConfig(
    level=logging.INFO,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Customize output format
)
logger = logging.getLogger(__name__)

DATASET_2_ICDV_MAP = {
    'sic': 10,
    'miiv': 10,
    'eicu': 10,
}

# ALL_FEATURES=("icds","ts_eos","ts_po2","static_adm","ts_epi_dur","ts_pt","static_age","ts_epi_rate","ts_ptt","static_bmi","ts_esr","ts_qsofa","static_height","ts_fgn","ts_rass","static_los_hosp","ts_fio2","ts_rbc","static_los_icu","ts_gcs","ts_rdw","static_sex","ts_glu","ts_resp","static_weight","ts_hba1c","ts_safi","ts_abx","ts_hct","ts_samp","ts_adh_rate","ts_hgb","ts_sbp","ts_alb","ts_hr","ts_sep3","ts_alp","ts_inr_pt","ts_sirs","ts_alt","ts_ins","ts_sofa_cardio","ts_ast","ts_k","ts_sofa_cns","ts_avpu","ts_lact","ts_sofa_coag","ts_basos","ts_lymph","ts_sofa","ts_be","ts_map","ts_sofa_liver","ts_bicar","ts_mchc","ts_sofa_renal","ts_bili","ts_mch","ts_sofa_resp","ts_bili_dir","ts_mcv","ts_supp_o2","ts_bnd","ts_methb","ts_susp_inf","ts_bun","ts_mews","ts_tco2","ts_ca","ts_mgcs","ts_temp","ts_cai","ts_mg","ts_tgcs","ts_ck","ts_na","ts_tnt","ts_ckmb","ts_neut","ts_tri","ts_cl","ts_news","ts_urine24","ts_crea","ts_norepi60","ts_urine","ts_crp","ts_norepi_dur","ts_vaso_ind","ts_dbp","ts_norepi_equiv","ts_vent_end","ts_death","ts_norepi_rate","ts_vent_start","ts_dobu60","ts_o2sat","ts_vgcs","ts_dobu_dur","ts_pafi","ts_wbc","ts_dobu_rate","ts_pco2","ws_dex","ts_dopa60","ts_ph","ws_ett_gcs","ts_dopa_dur","ts_phn_rate","ws_vent_ind","ts_dopa_rate","ts_phos","ts_egcs","ts_plt",)
ALL_FEATURES=(
    # "icds",
    "static_los_icu","ts_eos","ts_po2","static_adm","ts_epi_dur","ts_pt","static_age","ts_epi_rate","ts_ptt","static_bmi","ts_esr","ts_qsofa","static_height","ts_fgn","ts_rass","static_los_hosp","ts_fio2","ts_rbc","ts_gcs","ts_rdw","static_sex","ts_glu","ts_resp","static_weight","ts_hba1c","ts_safi","ts_abx","ts_hct","ts_samp","ts_adh_rate","ts_hgb","ts_sbp","ts_alb","ts_hr","ts_sep3","ts_alp","ts_inr_pt","ts_sirs","ts_alt","ts_ins","ts_sofa_cardio","ts_ast","ts_k","ts_sofa_cns","ts_avpu","ts_lact","ts_sofa_coag","ts_basos","ts_lymph","ts_sofa","ts_be","ts_map","ts_sofa_liver","ts_bicar","ts_mchc","ts_sofa_renal","ts_bili","ts_mch","ts_sofa_resp","ts_bili_dir","ts_mcv","ts_supp_o2","ts_bnd","ts_methb","ts_susp_inf","ts_bun","ts_mews","ts_tco2","ts_ca","ts_mgcs","ts_temp","ts_cai","ts_mg","ts_tgcs","ts_ck","ts_na","ts_tnt","ts_ckmb","ts_neut","ts_tri","ts_cl","ts_news","ts_urine24","ts_crea","ts_norepi60","ts_urine","ts_crp","ts_norepi_dur","ts_vaso_ind","ts_dbp","ts_norepi_equiv","ts_vent_end","ts_death","ts_norepi_rate","ts_vent_start","ts_dobu60","ts_o2sat","ts_vgcs","ts_dobu_dur","ts_pafi","ts_wbc","ts_dobu_rate","ts_pco2","ws_dex","ts_dopa60","ts_ph","ws_ett_gcs","ts_dopa_dur","ts_phn_rate","ws_vent_ind","ts_dopa_rate","ts_phos","ts_egcs","ts_plt",)
sALL_FEATURES = [feat for feat in ALL_FEATURES if feat != "icds" and 'ws_' not in feat]

PRIMARY_KEY_MAP = {
    "eicu": "patientunitstayid",
    "mimic": "patientunitstayid",
    "miiv": "stay_id",
    # "miiv": "icustay_id",
    "eicu_demo": "patientunitstayid",
    "mimic_demo": "icustay_id",
    "hirid": "patientid",
    "sic": "CaseID"
}
TIME_VARS_MAP_DATASET_LIST_KEYS = {
    "eicu": ["infusionoffset", "labresultoffset", "nursingchartoffset", "observationoffset","respchartoffset","respcarestatusoffset", "intakeoutputoffset", "culturetakenoffset", "drugstartoffset",  "index_var"],
    "eicu_demo": ["infusionoffset", "labresultoffset", "nursingchartoffset", "observationoffset","respchartoffset", "respcarestatusoffset", "intakeoutputoffset", "culturetakenoffset", "drugstartoffset",  "index_var"],
    "mimic": ['charttime', 'chartdate', 'index_var', 'startdate', 'starttime'],
    "mimic_demo": ['charttime', 'chartdate', 'index_var', 'startdate', 'starttime'],
    "hirid": ["givenat","datetime", "index_var"],
    "miiv": ["starttime","charttime","index_var","chartdate"],
    "sic": ["OffsetOfDeath","Offset","index_var"],	
}

LABEL_KEY_MAP = {
    "eicu": "icd9code",
    "mimic": "icd9_code",
    "miiv": "icd_code",
    "eicu_demo": "icd9code",
    "mimic_demo": "icd9_code",
    "sic": "ICD10Main"
}

non_ordinal = {'sex', 'adm',}
ordinal_variables = { 'avpu', }




# Define the filtering functions
feature_filters = {
    "static_age": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 20) & (x <= 90)]
    },
    "static_height": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 120) & (x <= 250)]
    },
    "static_weight": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 25) & (x <= 250)]
    },
    "ts_alp": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 0) & (x <= 250)]
    },
    "static_los_hosp": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 1) & (x <= 295)]
    },
    "static_los_icu": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 1) & (x <= 30)],
        "transform_fn": [lambda x, d: x / (3600 * 24) if d == 'sic' else x]
    },
    "ts_alb": {
        "type": "float",
        "filter_fn": [lambda x, d: (x > 0) & (x < 5)]
    },
    "ts_adh_rate": {
        "type": "float",
        "filter_fn": [lambda x, d: (x > 0) & (x < 0.169)]
    },
    "ts_ast": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 1000]
    },
    "ts_alt": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 1000],
    },
    "ts_bili_dir": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 35]
    },
    "ts_pt": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 250]
    },
    "ts_bili": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 60]
    },
    "ts_ck": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 10 * x.mean() if len(x) > 0 else True]
    },
    "ts_ckmb": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 500]
    },
    "ts_epi60": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 10000]
    },
    "ts_epi_dur": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 200]
    },
    "ts_ins": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 60]
    },
    "ts_mews": {
        "type": "float",
        "filter_fn": [lambda x, d: x.apply(lambda v: isinstance(v, int))]
    },
    "ts_phn_rate": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 50]
    },
    "ts_norepi_equiv": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 2]
    },
    "ts_norepi_dur": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 500]
    },
    "ts_methb": {
        "type": "float",
        "filter_fn": [lambda x, d: x < 40]
    },
    "ts_mch": {
        "type": "float",
        "filter_fn": [lambda x, d: (x > 15) & (x < 60)]
    },
}

enumerated_features = list(enumerate(["abx","adh_rate","adm","age","alb","alp","alt","ast","avpu","basos","be","bicar","bili","bili_dir","bmi","bnd","bun","ca","cai","ck","ckmb","cl","cort","crea","crp","dbp","death","dex","dobu_dur","dobu_rate","dobu60","dopa_dur","dopa_rate","dopa60","egcs","eos","epi_dur","epi_rate","epi60","esr","etco2","ett_gcs","fgn","fio2","gcs","glu","hba1c","hbco","hct","height","hgb","hr","inr_pt","ins","k","lact","los_hosp","los_icu","lymph","map","mch","mchc","mcv","mech_vent","methb","mews","mg","mgcs","na","neut","news","norepi_dur","norepi_equiv","norepi_rate","norepi60","o2sat","pafi","pco2","ph","phn_rate","phos","plt","po2","pt","ptt","qsofa","rass","rbc","rdw","resp","safi","samp","sbp","sep3","sex","sirs","sofa","sofa_cardio","sofa_cns","sofa_coag","sofa_liver","sofa_renal","sofa_resp","supp_o2","susp_inf","tco2","temp","tgcs","tnt","tri","urine","urine24","vaso_ind","vent_end","vent_ind","vent_start","vgcs","wbc","weight"]))



# Define the filtering functions
feature_filters = {
    "age": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 20) & (x <= 90)]
    },
    "height": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 120) & (x <= 250)]
    },
    "weight": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 25) & (x <= 250)]
    },
    "ts_alp": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 0) & (x <= 250)]
    },
    "los_hosp": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 1) & (x <= 295)]
    }, "los_icu": {
        "type": "float",
        "filter_fn": [lambda x, d: (x >= 1) & (x <= 30)],
        "transform_fn": [lambda x, d: x/(3600*24) if d == 'sic' else x]
    },
    "alb": {
        "type": "float",
         "filter_fn": [lambda x, d: (x > 0) & (x < 5)]
    },
    "adh_rate": {
        "type": "float",
         "filter_fn": [lambda x, d: (x > 0) & (x <0.169)]
    },
    "ast": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 10000]
    },
    "alt": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 1000],
        #  "transform_fn": [lambda x, d: x/1000]
    },
    "ast": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 1000],
        #  "transform_fn": [lambda x, d: x/1000]
    },
    "bili_dir": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 35]
    },
    "pt": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 250]
    },
    "bili": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 60]
    },
    "ck": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 10 * x.mean() if len(x) > 0 else True]
    },
    "ckmb": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 500 ]
    },
    "epi60": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 10000]
    }, 
    "epi_dur": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 200]
    },
    "ins": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 60]
    },
    "mews": {
        "type": "float",
         "filter_fn": [lambda x, d: x.apply(lambda v: isinstance(v, int))]
    },
    "phn_rate": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 50]
    },
    "norepi_equiv": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 2]
    },
    "norepi_dur": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 500]
    },
    "methb": {
        "type": "float",
         "filter_fn": [lambda x, d: x < 40]
    },
    "mch": {
        "type": "float",
         "filter_fn": [lambda x, d: (x > 15) & (x < 60)]
    },
}


categorical_features = {
    "adm": 'one_hot',
    "avpu": ['A','V','P','U'], # ordinal AVPU (an acronym for Alert, Voice, Pain, Unresponsive) is a simple assessment scale to access the conscious level of residents.
    "sex": 'boolean', 
    "abx": 'boolean', # TODO add negation 
    "death": 'boolean', # TODO add negation if an id misses 
    "samp": 'boolean', 
    "sep3": 'boolean', # TODO rest false 0 ,  #  sepsis-3 criterion  
    "susp_inf": 'boolean', #  TODO suspected infection rest 0  
    "supp_o2":  'boolean',  # it contains both values
    "vaso_ind": 'boolean',   #  TODO  put False the rest 0 #    vasopressor indicator Vasopressors and inotropes are medications used to create vasoconstriction or increase cardiac contractility, respectively, in patients with shock or any other reason for extremely low blood press
    "vent_end": 'boolean',  # TODO default 0
    "vent_start": 'boolean', # TODO default 0
    "vent_ind": 'boolean',  # TODO default  0  it is ordinal 
    "mech_vend": 'boolean', # TODO third continus # TODO boolean last only 
    "ett_gcs": 'boolean', # rest 0
}  


def is_time_key(col, tkeys):
    for tkey in tkeys:
        if tkey in col:
            return True
    return False

def find_pkeys_per_dataset(INPUT_DIR, dataset_name, pkey_column):
    pkeys = set()  # Set to store unique pkeys
    try:
        lf=os.listdir(INPUT_DIR)
    except:
        import pdb; pdb.set_trace()
    assert len(lf) > 0, f"No files found in {INPUT_DIR}"
    for feature in tqdm(lf):
        df = pd.read_csv(f'{INPUT_DIR}/{feature}')
        print(f'Processing {feature} for {dataset_name}')
        pkeys.update(df[pkey_column].unique())
        # pkeys as list 
    pkeys = list(pkeys)
    return pkeys

def find_last_timestep_per_pkey(INPUT_DIR, dataset_name, tkeys, pkey_column):
    "Based on Lenght of Stay (LoS) ids we keep the last timestep for which time is imputed."
    los_icu_key = 'los_icu'
    fpath = f'{INPUT_DIR}/{dataset_name}_static_{los_icu_key}.csv'
    df = pd.read_csv(fpath)
    if pkey_column not in df.columns:
        print(f"Error: {dataset_name} {fpath}")
    # if los icu na drop 
    df.dropna(inplace=True)
    los_icu_key = df.columns[-1]
    # use first col for key val for los icu 
    last_timesteps = dict(zip(df[pkey_column], df[los_icu_key]))

    return last_timesteps
            
        

def get_feature_name_from_file(file: str) -> str:
    """
    Extracts the feature name including the prefix (e.g., 'ts_ptt') from the given file path.
    
    Examples:
        - "data/step_2_mimic_demo/293178/mimic_demo_ts_ptt.csv" -> "ts_ptt"
        - "data/step_2_mimic_demo/293178/miiv_ws_blood_pressure.csv" -> "ws_blood_pressure"
    
    Args:
        file (str): The file path.
    
    Returns:
        str: The extracted feature name, e.g., 'ts_ptt'.
    
    Raises:
        ValueError: If the feature name cannot be found.
    """
    # Extract the filename from the path
    filename = os.path.basename(file)
    
    # Regex pattern to capture 'ts_ptt', 'st_ptt', or 'ws_ptt'
    match = re.search(r'((?:static|ts|ws)_\w+)\.csv$', filename)
    type_of_feature = re.search(r'(static|ts|ws)_', filename)
    if match:
        feature_name = match.group(1) 
        # remove ts_ static_ ws_ from feature name
        feature_name = feature_name.replace('static_', '').replace('ts_', '').replace('ws_', '')
        return feature_name, type_of_feature.group(1)
    
    raise ValueError(f"Feature name not found in file: {file}")

assert get_feature_name_from_file("data/step_2_mimic_demo/293178/mimic_demo_ts_ptt.csv")[0] == "ptt"
assert get_feature_name_from_file("data/step_2_mimic_demo/293178/mimic_demo_ts_ptt.csv")[1] == "ts"
assert get_feature_name_from_file("data/step_2_miiv/293178/miiv_ts_ptt.csv")[0] == "ptt"
assert get_feature_name_from_file("data/step_2_miiv/293178/miiv_ts_ptt.csv")[1] == "ts"


def split_bysubject_miiv():
    # Read the CSV file
    df = pd.read_csv('data/stays_miiv.csv') # TODO  need to downklaod in beginning in step1 
    # Group by subject_id and collect hadm_id into a list
    grouped = df.groupby('subject_id')['stay_id'].apply(list).to_dict()
    with open('data/miiv_grouped.json', 'w') as f: json.dump(grouped, f)

    print('saved grouped data to data/miiv_grouped.json')
    # read test 
    with open('data/miiv_grouped.json', 'r') as f: grouped1 = json.load(f)

if False: # s need to be executed for miiv - TODO put in data_specific scriptins 
    if not os.path.exists('data/miiv_grouped.json'):
        split_bysubject_miiv()

    with open('data/miiv_grouped.json', 'r') as f: subject_to_icustay = json.load(f)


# Define ICD-9 chapters with their code ranges
icd_9_cm_chapters = [
    ("Infectious And Parasitic Diseases", 1, 139),
    ("Neoplasms", 140, 239),
    ("Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders", 240, 279),
    ("Diseases Of The Blood And Blood-Forming Organs", 280, 289),
    ("Mental Disorders", 290, 319),
    ("Diseases Of The Nervous System And Sense Organs", 320, 389),
    ("Diseases Of The Circulatory System", 390, 459),
    ("Diseases Of The Respiratory System", 460, 519),
    ("Diseases Of The Digestive System", 520, 579),
    ("Diseases Of The Genitourinary System", 580, 629),
    ("Complications Of Pregnancy, Childbirth, And The Puerperium", 630, 679),
    ("Diseases Of The Skin And Subcutaneous Tissue", 680, 709),
    ("Diseases Of The Musculoskeletal System And Connective Tissue", 710, 739),
    ("Congenital Anomalies", 740, 759),
    ("Certain Conditions Originating In The Perinatal Period", 760, 779),
    ("Symptoms, Signs, And Ill-Defined Conditions", 780, 799),
    ("Injury And Poisoning", 800, 999),
    ("Supplementary Classification Of External Causes Of Injury And Poisoning", 800, 999)
]

fpaths_lstm={
    'mimic_demo': 'data/embeddings/singleLSTM_embeddings_mimic_demo_samples_1000__ms_2.pkl',
    'sic':'data/embeddings/singleLSTM_embeddings_sic_samples_6000__ms_6000.dat'
}