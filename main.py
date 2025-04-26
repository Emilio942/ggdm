import os
import os.path as osp
import tarfile
import requests
# import gzip # No longer explicitly needed
import shutil
import logging
import traceback
import hashlib
import json
from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.typing import Tensor
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
# from rdkit.Chem import Descriptors # Not used in the final feature set yet
from rdkit import RDLogger
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process

# --- Configuration for Limited Logging ---
# Maximum number of times a specific warning message will be logged
MAX_LOG_PER_WARNING_TYPE = 5
# Dictionary to store warning counts (will be global per worker process)
# Using defaultdict to simplify counting
warning_counters: Dict[str, int] = defaultdict(int)

# --- Basic Logging Setup ---
# Keep DEBUG level to catch initial errors, but limit repetitive warnings below
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
RDLogger.DisableLog('rdApp.*')
logging.info("RDKit Logger disabled.")


# --- Helper for Limited Logging ---
def log_limited_warning(message: str, key: Optional[str] = None):
    """Logs a warning message only up to MAX_LOG_PER_WARNING_TYPE times per key."""
    global warning_counters, MAX_LOG_PER_WARNING_TYPE
    log_key = key if key else message # Use message as key if no specific key provided
    
    warning_counters[log_key] += 1
    
    if warning_counters[log_key] <= MAX_LOG_PER_WARNING_TYPE:
        logging.warning(message)
    elif warning_counters[log_key] == MAX_LOG_PER_WARNING_TYPE + 1:
        # Log suppression message only once per key
        logging.warning(f"Further warnings of type '{log_key}' suppressed...")

# --- Configuration Dataclass ---
@dataclass
class QM9Config:
    """Configuration for QM9 Dataset processing."""
    # Core settings
    root: str = osp.join(os.getcwd(), 'data', 'QM9_enhanced')
    version_name: str = "v1" # Manual version name for processed data structure

    # Feature Extraction
    use_h_atoms: bool = True # Recommended for QM9 properties. If False, hydrogens are removed.
    allowed_atom_symbols: List[str] = field(default_factory=lambda: ['H', 'C', 'N', 'O', 'F'])
    allowed_bond_types: List[BT] = field(default_factory=lambda: [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC])

    # --- Optional Atom Features ---
    add_atomic_mass: bool = False
    add_formal_charge: bool = False
    add_hybridization: bool = False
    add_is_aromatic_atom: bool = False
    add_is_in_ring_atom: bool = False

    # --- Optional Bond Features ---
    add_is_conjugated_bond: bool = False
    add_is_in_ring_bond: bool = False

    # Target Properties
    all_target_properties: List[str] = field(default_factory=lambda: [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ])
    target_keys_to_load: Optional[List[str]] = None # If None, load all_target_properties

    # Processing & Download
    qm9_url: str = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
    raw_archive_name: str = 'gdb9.tar.gz'
    raw_sdf_name: str = 'gdb9.sdf'
    force_download: bool = False
    force_process: bool = False
    num_workers: int = max(1, cpu_count() // 2) # Use half the CPU cores for processing

    # Data Splits (optional)
    split_definition_path: Optional[str] = None # Path to a .pt file with {'train_idx': tensor, 'val_idx': tensor, 'test_idx': tensor}
    load_split: Optional[str] = None # 'train', 'val', or 'test'. If None, load all data.

    # Validation
    check_duplicates: bool = True # Check for duplicates using canonical SMILES

    # --- Internal/Calculated ---
    _atom_symbol_map: Dict[str, int] = field(init=False, repr=False)
    _bond_type_map: Dict[BT, int] = field(init=False, repr=False)
    _actual_target_keys: List[str] = field(init=False, repr=False)
    _config_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        """Calculate derived attributes after initialization."""
        self._atom_symbol_map = {symbol: i for i, symbol in enumerate(self.allowed_atom_symbols)}
        self._bond_type_map = {bond_type: i for i, bond_type in enumerate(self.allowed_bond_types)}

        if self.target_keys_to_load is None:
            self._actual_target_keys = self.all_target_properties
        else:
            invalid_keys = [k for k in self.target_keys_to_load if k not in self.all_target_properties]
            if invalid_keys:
                 raise ValueError(f"Invalid target keys requested: {invalid_keys}. Allowed: {self.all_target_properties}")
            self._actual_target_keys = self.target_keys_to_load
        logging.info(f"Will load target properties: {self._actual_target_keys}")

        # --- Generate a hash based on relevant config options for versioning ---
        hash_relevant_field_names = [
            'use_h_atoms', 'allowed_atom_symbols', 'allowed_bond_types',
            'add_atomic_mass', 'add_formal_charge', 'add_hybridization',
            'add_is_aromatic_atom', 'add_is_in_ring_atom',
            'add_is_conjugated_bond', 'add_is_in_ring_bond',
            'all_target_properties', 'target_keys_to_load', 'check_duplicates'
        ]
        hash_relevant_config = {
            k: getattr(self, k) for k in hash_relevant_field_names if hasattr(self, k)
        }
        config_str = json.dumps(hash_relevant_config, sort_keys=True, default=str)
        self._config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        # --- End Hash Calculation ---

        logging.info(f"Configuration hash for versioning: {self._config_hash}")

        # Validate split options
        if self.load_split and not self.split_definition_path:
            raise ValueError("`load_split` requires `split_definition_path` to be set.")
        if self.load_split and self.load_split not in ['train', 'val', 'test']:
            raise ValueError("`load_split` must be one of 'train', 'val', 'test' or None.")

# --- Helper Function: Molecule to PyG Data (now using Config and Limited Logging) ---

def rdkit_mol_to_pyg_data_configured(mol: Chem.Mol, config: QM9Config) -> Optional[Data]:
    """
    Converts an RDKit molecule to PyG Data using settings from QM9Config.
    Uses canonical SMILES and limits repetitive warnings.
    """
    canonical_smiles = "[Unavailable]"
    try:
        if mol is None: return None
        if mol.GetNumAtoms() == 0: return None # Should be caught earlier, but safe check

        try:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            canonical_smiles = "[Canonical SMILES Generation Failed]"

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            # Use limited logging for repetitive warnings
            log_limited_warning(f"Molecule {canonical_smiles} has 0 atoms after potential processing. Skipping.", key="zero_atoms")
            return None

        # --- Get Conformer Positions ---
        try:
            conformer = mol.GetConformer()
            positions = conformer.GetPositions()
            pos_tensor = torch.tensor(positions, dtype=torch.float)
            if pos_tensor.shape[0] != num_atoms:
                 log_limited_warning(f"Skipping {canonical_smiles}: Mismatch num_atoms ({num_atoms}) vs conformer positions ({pos_tensor.shape[0]}).", key="pos_mismatch")
                 return None
        except (AttributeError, ValueError) as e:
            log_limited_warning(f"Skipping {canonical_smiles}: Error getting conformer/positions: {e}", key="conformer_error")
            return None

        # --- Atom Features ---
        atom_features_list = []
        atomic_numbers = []
        for i, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in config._atom_symbol_map:
                log_limited_warning(f"Skipping {canonical_smiles}: Unsupported atom type '{atom_symbol}'. Allowed: {config.allowed_atom_symbols}", key=f"unsupported_atom_{atom_symbol}")
                return None

            atom_type_onehot = torch.zeros(len(config.allowed_atom_symbols), dtype=torch.float)
            atom_type_onehot[config._atom_symbol_map[atom_symbol]] = 1.0
            features = [atom_type_onehot]
            atomic_numbers.append(atom.GetAtomicNum())

            # Optional features... (unchanged logic)
            if config.add_atomic_mass: features.append(torch.tensor([atom.GetMass()], dtype=torch.float))
            if config.add_formal_charge: features.append(torch.tensor([atom.GetFormalCharge()], dtype=torch.float))
            if config.add_hybridization:
                hyb = atom.GetHybridization(); hyb_map = {Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1, Chem.HybridizationType.SP3: 2, Chem.HybridizationType.SP3D: 3, Chem.HybridizationType.SP3D2: 4}
                hyb_onehot = torch.zeros(5, dtype=torch.float);
                if hyb in hyb_map: hyb_onehot[hyb_map[hyb]] = 1.0
                features.append(hyb_onehot)
            if config.add_is_aromatic_atom: features.append(torch.tensor([1.0 if atom.GetIsAromatic() else 0.0], dtype=torch.float))
            if config.add_is_in_ring_atom: features.append(torch.tensor([1.0 if atom.IsInRing() else 0.0], dtype=torch.float))

            atom_features_list.append(torch.cat(features))

        x_tensor = torch.stack(atom_features_list, dim=0)

        # --- Edge Features & Indices ---
        edge_indices_list = []
        edge_features_list = []
        if num_atoms > 1:
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                if bond_type not in config._bond_type_map:
                    log_limited_warning(f"Skipping {canonical_smiles}: Unsupported bond type '{bond_type}'. Allowed: {config.allowed_bond_types}", key=f"unsupported_bond_{bond_type}")
                    return None

                start_atom_idx, end_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type_onehot = torch.zeros(len(config.allowed_bond_types), dtype=torch.float)
                bond_type_onehot[config._bond_type_map[bond_type]] = 1.0
                features = [bond_type_onehot]

                # Optional features... (unchanged logic)
                if config.add_is_conjugated_bond: features.append(torch.tensor([1.0 if bond.GetIsConjugated() else 0.0], dtype=torch.float))
                if config.add_is_in_ring_bond: features.append(torch.tensor([1.0 if bond.IsInRing() else 0.0], dtype=torch.float))

                edge_feature_vector = torch.cat(features)
                edge_indices_list.extend([(start_atom_idx, end_atom_idx), (end_atom_idx, start_atom_idx)])
                edge_features_list.extend([edge_feature_vector, edge_feature_vector])

        num_edges = len(edge_indices_list)
        if num_edges > 0:
            edge_index_tensor = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.stack(edge_features_list, dim=0)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            expected_edge_dim = len(config.allowed_bond_types)
            if config.add_is_conjugated_bond: expected_edge_dim += 1
            if config.add_is_in_ring_bond: expected_edge_dim += 1
            edge_attr_tensor = torch.empty((0, expected_edge_dim), dtype=torch.float)

        # --- Target Properties ---
        target_values = []
        mol_props = mol.GetPropsAsDict()
        for key in config._actual_target_keys:
            try:
                prop_value = float(mol_props[key])
                target_values.append(prop_value)
            except KeyError:
                log_limited_warning(f"Skipping {canonical_smiles}: Target property '{key}' not found.", key=f"missing_prop_{key}")
                return None
            except ValueError:
                prop_val_str = mol_props.get(key, '[N/A]')
                log_limited_warning(f"Skipping {canonical_smiles}: Could not convert property '{key}' value '{prop_val_str}' to float.", key=f"convert_prop_error_{key}")
                return None

        y_tensor = torch.tensor(target_values, dtype=torch.float).unsqueeze(0)

        # --- Create PyG Data Object ---
        data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor,
                    pos=pos_tensor, y=y_tensor, smiles=canonical_smiles,
                    atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
                    num_nodes=num_atoms)

        # --- Final Validation ---
        validation_result = data.validate(raise_on_error=False)
        if not validation_result:
             errors = getattr(data, 'validation_errors', 'N/A')
             # Use limited logging here as well
             log_limited_warning(f"Data validation failed for {canonical_smiles}. Skipping. Errors: {errors}", key="validation_error")
             return None

        return data

    except Exception as e:
        # Log critical unexpected errors fully, but maybe only once per type if needed
        log_key = f"unexpected_error_{type(e).__name__}"
        if warning_counters[log_key] < 1: # Log unexpected errors fully at least once
             logging.error(f"Unexpected error processing molecule (SMILES: {canonical_smiles}): {e}")
             logging.debug(traceback.format_exc()) # Keep traceback for debug level
             warning_counters[log_key] += 1
        elif warning_counters[log_key] == 1:
             logging.error(f"Further unexpected errors of type {type(e).__name__} suppressed...")
             warning_counters[log_key] += 1
        # Still return None on any exception
        return None

# --- Worker function for multiprocessing ---
_GLOBAL_CONFIG = None
_GLOBAL_SDF_PATH = None

def init_worker(config_dict: dict, sdf_path: str):
    """Initializer for multiprocessing worker pool. Resets warning counters."""
    global _GLOBAL_CONFIG, _GLOBAL_SDF_PATH, warning_counters
    # Reset counters for this specific worker process
    warning_counters = defaultdict(int)
    try:
        _GLOBAL_CONFIG = QM9Config(**config_dict)
        _GLOBAL_SDF_PATH = sdf_path
        RDLogger.DisableLog('rdApp.*')
        # logging.debug(f"Worker {current_process().name} initialized.")
    except Exception as e:
        # Log the critical initialization error fully, it should not happen often
        logging.error(f"Error initializing worker {current_process().name}: {e}")
        logging.exception("Traceback during worker initialization:")
        _GLOBAL_CONFIG = None
        _GLOBAL_SDF_PATH = None

def process_mol_worker(mol_index: int) -> Optional[Data]:
    """Processes a single molecule by index in a worker process."""
    global _GLOBAL_CONFIG, _GLOBAL_SDF_PATH
    if _GLOBAL_CONFIG is None or _GLOBAL_SDF_PATH is None:
        # Do not flood logs if init failed repeatedly for this worker
        log_limited_warning(f"Worker {current_process().name} not initialized properly. Skipping task for index {mol_index}.", key="worker_init_failed")
        return None

    try:
        supplier = Chem.SDMolSupplier(_GLOBAL_SDF_PATH, removeHs=(not _GLOBAL_CONFIG.use_h_atoms), sanitize=True)
        # Check index bounds robustly
        # Getting len(supplier) can be slow; iterating might be better but complex for index access.
        # Let's rely on exception handling for invalid index.
        mol = supplier[mol_index]
        del supplier
    except IndexError:
         log_limited_warning(f"Worker {current_process().name}: Index {mol_index} out of bounds.", key="index_out_of_bounds")
         return None
    except Exception as e:
         log_limited_warning(f"Worker {current_process().name}: Error reading index {mol_index}: {e}", key="supplier_read_error")
         return None

    if mol is None:
        # log_limited_warning(f"Worker {current_process().name}: RDKit failed parse index {mol_index}.", key="rdkit_parse_error")
        # This can be very common if SDF is noisy, might omit warning entirely or keep it very limited
        pass # Optionally skip logging this common case to reduce noise further
        return None

    # Processing function now handles limited logging internally
    return rdkit_mol_to_pyg_data_configured(mol, _GLOBAL_CONFIG)

# --- Enhanced QM9 Dataset Class ---
class QM9EnhancedDataset(InMemoryDataset):
    """
    Enhanced QM9 Dataset Loader with limited logging for repetitive warnings.
    """
    def __init__(self, config: QM9Config, transform=None, pre_transform=None, pre_filter=None):
        self.config = config
        self._raw_sdf_path = osp.join(self.config.root, 'raw', self.config.raw_sdf_name)
        self._processed_dir_path = osp.join(self.config.root, 'processed', self.config.version_name)
        # Reset global counter for the main process duplicate check (if needed)
        self._main_process_duplicate_counter = 0
        logging.info(f"Initializing QM9EnhancedDataset in root: {self.config.root}...")
        # ... (rest of __init__ including super call and split logic remains the same)
        super().__init__(self.config.root, transform, pre_transform, pre_filter)
        try:
             self.data, self.slices = torch.load(self.processed_paths[0])
             logging.info(f"Dataset loaded successfully from {self.processed_paths[0]}")
        except FileNotFoundError:
             logging.error(f"Processed file not found at {self.processed_paths[0]}. Ensure processing ran.")
             raise
        except Exception as e:
            logging.error(f"Error loading processed file {self.processed_paths[0]}: {e}")
            raise
        if self.config.load_split: self._apply_split()

    @property
    def raw_dir(self) -> str: return osp.join(self.config.root, 'raw')
    @property
    def processed_dir(self) -> str: os.makedirs(self._processed_dir_path, exist_ok=True); return self._processed_dir_path
    @property
    def raw_file_names(self) -> List[str]: return [self.config.raw_sdf_name]
    @property
    def processed_file_names(self) -> List[str]: return [f'qm9_pyg_{self.config._config_hash}.pt']

    def download(self):
        """Downloads and extracts the QM9 SDF file."""
        # (Download logic remains largely the same - robust version from before)
        os.makedirs(self.raw_dir, exist_ok=True)
        raw_archive_path = osp.join(self.raw_dir, self.config.raw_archive_name)
        if osp.exists(self._raw_sdf_path) and not self.config.force_download:
            logging.info(f"Raw file exists: {self._raw_sdf_path}. Skipping download.")
            return
        if osp.exists(self._raw_sdf_path) and self.config.force_download:
             logging.info(f"Forcing download, removing existing: {self._raw_sdf_path}")
             try: os.remove(self._raw_sdf_path)
             except OSError as e: logging.warning(f"Could not remove {self._raw_sdf_path}: {e}")
        logging.info(f"Downloading {self.config.qm9_url} to {raw_archive_path}...")
        try:
            response = requests.get(self.config.qm9_url, stream=True, timeout=60); response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {self.config.raw_archive_name}", leave=False) as pbar:
                with open(raw_archive_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk); pbar.update(len(chunk))
            if total_size != 0 and pbar.n != total_size: raise IOError("Download incomplete.")
            logging.info("Download complete.")
            logging.info(f"Extracting {self.config.raw_sdf_name}...")
            with tarfile.open(raw_archive_path, 'r:gz') as tar:
                 member_info = next((m for m in tar.getmembers() if m.name.endswith(self.config.raw_sdf_name)), None)
                 if member_info is None: raise FileNotFoundError("SDF file not in archive.")
                 member_info.name = osp.basename(member_info.name); tar.extract(member_info, path=self.raw_dir)
                 if not osp.exists(self._raw_sdf_path): raise FileNotFoundError("Extracted file not found.")
            logging.info(f"Extraction complete to {self._raw_sdf_path}.")
            try: os.remove(raw_archive_path); logging.info(f"Removed archive.")
            except OSError as e: logging.warning(f"Could not remove archive: {e}")
        except Exception as e:
            logging.error(f"Download/Extraction failed: {e}")
            if osp.exists(raw_archive_path): try: os.remove(raw_archive_path)
            except OSError: pass
            if osp.exists(self._raw_sdf_path): try: os.remove(self._raw_sdf_path)
            except OSError: pass
            raise RuntimeError("Failed to download or extract QM9 data.") from e

    def process(self):
        """Processes the raw SDF file with limited logging for warnings."""
        logging.info(f"Starting parallel processing (up to {self.config.num_workers} workers)...")
        start_time = time.time()

        logging.info(f"Counting molecules in {self._raw_sdf_path}...")
        try:
             # Safely count molecules
             counter_supplier = Chem.SDMolSupplier(self._raw_sdf_path, removeHs=(not self.config.use_h_atoms), sanitize=True)
             num_molecules = sum(1 for _ in counter_supplier); del counter_supplier
             logging.info(f"Counted {num_molecules} molecule entries.")
             if num_molecules == 0: raise ValueError("SDF contains no processable entries.")
        except Exception as e:
            logging.error(f"Failed to count molecules: {e}"); raise

        init_args = (asdict(self.config), self._raw_sdf_path)
        data_list = []
        skipped_total = 0 # Count all skips (errors, duplicates, filters)
        processed_count = 0
        seen_smiles: Set[str] = set()
        duplicate_count = 0

        try:
            actual_workers = min(self.config.num_workers, num_molecules)
            if actual_workers < self.config.num_workers: logging.info(f"Using {actual_workers} workers.")

            with Pool(processes=actual_workers, initializer=init_worker, initargs=init_args) as pool:
                results_iterator = pool.imap_unordered(process_mol_worker, range(num_molecules))
                with tqdm(total=num_molecules, desc="Processing Molecules (Parallel)", smoothing=0.1) as pbar:
                    for i, result_data in enumerate(results_iterator):
                        pbar.update(1)
                        if result_data is not None:
                            # --- Duplicate Check (Main Process Counter) ---
                            if self.config.check_duplicates:
                                if result_data.smiles in seen_smiles:
                                    # Limit duplicate logging using the main process counter
                                    self._main_process_duplicate_counter += 1
                                    if self._main_process_duplicate_counter <= MAX_LOG_PER_WARNING_TYPE:
                                        logging.debug(f"Duplicate canonical SMILES detected and skipped: {result_data.smiles}")
                                    elif self._main_process_duplicate_counter == MAX_LOG_PER_WARNING_TYPE + 1:
                                        logging.debug("Further duplicate messages suppressed...")
                                    duplicate_count += 1
                                    skipped_total += 1
                                    continue
                                else:
                                    seen_smiles.add(result_data.smiles)

                            # --- Apply pre_filter/pre_transform ---
                            data_valid = True
                            if self.pre_filter is not None and not self.pre_filter(result_data):
                                skipped_total += 1; data_valid = False
                            if data_valid and self.pre_transform is not None:
                                try:
                                    transformed_data = self.pre_transform(result_data)
                                    if transformed_data is None:
                                        # Use limited logging for transform skips
                                        log_limited_warning(f"Molecule {getattr(result_data, 'smiles', '[N/A]')} became None after pre_transform. Skipping.", key="pre_transform_none")
                                        skipped_total += 1; data_valid = False
                                    else: result_data = transformed_data
                                except Exception as e:
                                    # Log pre_transform errors fully once
                                    log_key = f"pre_transform_error_{type(e).__name__}"
                                    if warning_counters[log_key] < 1: # Log fully once
                                        logging.error(f"Error pre_transform {getattr(result_data, 'smiles', '[N/A]')}: {e}")
                                        logging.debug(traceback.format_exc())
                                        warning_counters[log_key] += 1
                                    elif warning_counters[log_key] == 1:
                                        logging.error(f"Further pre_transform errors ({type(e).__name__}) suppressed...")
                                        warning_counters[log_key] += 1
                                    skipped_total += 1; data_valid = False

                            if data_valid:
                                data_list.append(result_data); processed_count += 1
                        else:
                            # Worker returned None (error reason logged by worker with limits)
                            skipped_total += 1
        except Exception as e:
             logging.error(f"Multiprocessing error: {e}"); logging.exception("Traceback:"); raise RuntimeError("MP pool error.")

        end_time = time.time(); duration = end_time - start_time
        logging.info(f"Parallel processing finished in {duration:.2f}s.")
        if duration > 0 and processed_count > 0: logging.info(f"Speed: {processed_count / duration:.2f} mols/sec.")

        if duplicate_count > 0: logging.info(f"Skipped {duplicate_count} duplicate molecules.")
        # Calculate other skips
        other_skips = skipped_total - duplicate_count
        if other_skips > 0: logging.warning(f"Skipped {other_skips} additional entries due to errors or filtering (see logs for first {MAX_LOG_PER_WARNING_TYPE} instances of each type).")

        if not data_list:
             logging.error("No molecules processed successfully.")
             logging.error(f"Counts: Initial={num_molecules}, Processed OK={processed_count}, Duplicates={duplicate_count}, Other Skips={other_skips}")
             raise ValueError("Processing resulted in an empty dataset.")

        logging.info(f"Successfully processed {processed_count} unique molecules.")
        logging.info(f"Collating {len(data_list)} data objects...")
        data, slices = self.collate(data_list)

        processed_path = self.processed_paths[0]
        logging.info(f"Saving processed data to {processed_path}...")
        try:
            torch.save((data, slices), processed_path)
            logging.info("Saving complete.")
            # --- Add Metadata Saving ---
            self._save_metadata(processed_path, processed_count, duplicate_count, other_skips, duration)
        except Exception as e:
             logging.error(f"Failed to save processed data: {e}")
             if osp.exists(processed_path): try: os.remove(processed_path) catch OSError: pass
             raise

    def _save_metadata(self, data_path: str, num_processed: int, num_duplicates: int, num_other_skips: int, duration: float):
        """Saves metadata alongside the processed data file."""
        metadata_path = Path(data_path).with_suffix('.json')
        logging.info(f"Saving metadata to {metadata_path}...")
        
        # Collect library versions
        try: import rdkit; rdkit_version = rdkit.__version__
        except ImportError: rdkit_version = "Not Found"
        try: import torch_geometric; pyg_version = torch_geometric.__version__
        except ImportError: pyg_version = "Not Found"
        try: import platform; python_version = platform.python_version()
        except: python_version = "Unknown"

        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "config": asdict(self.config, dict_factory=lambda x: {k: str(v) if isinstance(v, BT) else v for k, v in x}), # Convert BondType for JSON
            "processing_stats": {
                "num_processed_ok": num_processed,
                "num_duplicates_skipped": num_duplicates,
                "num_other_skips_errors": num_other_skips,
                "processing_duration_sec": round(duration, 2),
            },
            "versions": {
                "python": python_version,
                "pytorch": torch.__version__,
                "torch_geometric": pyg_version,
                "rdkit": rdkit_version,
            },
            "processed_data_file": osp.basename(data_path),
            "source_sdf_file": self.config.raw_sdf_name,
        }
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logging.info("Metadata saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save metadata to {metadata_path}: {e}")


    def _apply_split(self):
        """Filters loaded data based on split definition file."""
        # (Split logic remains the same - robust version from before)
        if not self.config.load_split: return
        if not self.config.split_definition_path: raise ValueError("Split path needed.")
        split_file = Path(self.config.split_definition_path);
        if not split_file.is_file(): raise FileNotFoundError(f"Split file not found: {split_file}")
        try:
            logging.info(f"Loading split indices from {split_file}..."); split_dict = torch.load(split_file)
            split_key = f"{self.config.load_split}_idx"; indices = split_dict.get(split_key)
            if indices is None: raise ValueError(f"Split key '{split_key}' not in {split_file}.")
            if not isinstance(indices, torch.Tensor): indices = torch.tensor(indices, dtype=torch.long)
            if indices.dtype != torch.long: indices = indices.to(torch.long)
            num_total, num_indices = self.len(), len(indices)
            logging.info(f"Applying split '{self.config.load_split}': {num_indices} indices (from {num_total}).")
            if num_indices > 0:
                 max_idx = indices.max().item()
                 if max_idx >= num_total: raise IndexError(f"Split index {max_idx} out of bounds ({num_total}).")
            self.data, self.slices = self.index_select(indices)
            logging.info(f"Dataset size after split: {self.len()}")
            if self.len() != num_indices and num_indices > 0: logging.warning("Final size != num indices.")
        except Exception as e: logging.error(f"Split failed: {e}"); raise RuntimeError("Split error.") from e

# --- Example Usage Script ---
if __name__ == '__main__':
    logging.info("--- Starting Enhanced QM9 Dataset Preparation Script ---")

    # --- Configuration ---
    # Use a configuration that might trigger some warnings initially
    config = QM9Config(
        root=osp.join(os.getcwd(), 'data', 'QM9_LimitLog'), # New root/version
        version_name="v1_limited_log_test",
        target_keys_to_load=['U0', 'gap', 'mu'], # Standard targets
        add_formal_charge=True,
        add_is_aromatic_atom=True,
        add_hybridization=True,
        check_duplicates=True,
        num_workers=max(1, cpu_count() // 2), # Use multiple workers again
        # --- Split Configuration (Still using Dummy for example) ---
        split_definition_path=osp.join(os.getcwd(), 'data', 'qm9_std_splits.pt'),
        load_split=None # Load all data for now to see processing
        # force_process=True # Force reprocessing to test logging limits
    )

    # --- Create Dummy Split File (if necessary) ---
    dummy_split_path = Path(config.split_definition_path) if config.split_definition_path else None
    if dummy_split_path and not dummy_split_path.exists():
         logging.warning(f"Creating DUMMY split file: {dummy_split_path}")
         os.makedirs(dummy_split_path.parent, exist_ok=True)
         num_total_qm9_approx=133885; num_val=10000; num_test=10000
         num_train=num_total_qm9_approx-num_val-num_test; indices=torch.randperm(num_total_qm9_approx)
         train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train+num_val], indices[num_train+num_val:]
         torch.save({'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}, dummy_split_path)
         logging.info(f"Dummy splits created: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    elif dummy_split_path: logging.info(f"Using existing split file: {dummy_split_path}")
    else: logging.info("No split file configured.")

    try:
        os.makedirs(config.root, exist_ok=True)
        logging.info("Instantiating QM9EnhancedDataset...")
        start_instantiation = time.time()
        # Reset main process warning counter for this run
        warning_counters = defaultdict(int)
        qm9_dataset = QM9EnhancedDataset(config=config)
        end_instantiation = time.time()
        logging.info(f"Dataset instantiation took {end_instantiation - start_instantiation:.2f} seconds.")

        logging.info("\n--- Dataset Inspection ---")
        logging.info(f"Dataset object: {qm9_dataset}")
        logging.info(f"Number of graphs loaded: {len(qm9_dataset)}")

        if len(qm9_dataset) > 0:
            first_data = qm9_dataset[0]
            logging.info("\n--- Inspecting First Data Point ---")
            # Keep inspection brief as requested
            logging.info(f"Node feature shape (x): {first_data.x.shape}")
            logging.info(f"Edge feature shape (edge_attr): {first_data.edge_attr.shape}")
            logging.info(f"Target properties shape (y): {first_data.y.shape}")
            logging.info(f"Canonical SMILES: {first_data.smiles}")
            # Verification checks (optional, can be verbose)
            # assert first_data.y.shape[1] == len(config._actual_target_keys)
            # ... other assertions ...
            logging.info("Basic inspection complete.")
        else:
            logging.warning("Dataset is empty after processing/loading.")

        logging.info("\n--- Enhanced QM9 Dataset script finished successfully! ---")

    except (FileNotFoundError, ValueError, RuntimeError, ConnectionError, AssertionError, IndexError) as e:
        logging.error(f"\n--- CRITICAL ERROR ---")
        logging.error(f"Error Type: {type(e).__name__}")
        logging.error(f"Details: {e}")
        logging.exception("Traceback:")
    except Exception as e:
        logging.error(f"\n--- UNEXPECTED ERROR ---")
        logging.error(f"Details: {e}")
        logging.exception("Traceback:")
        
