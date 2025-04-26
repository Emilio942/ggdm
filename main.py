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

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
RDLogger.DisableLog('rdApp.*')
logging.info("RDKit Logger disabled.")

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
        \"\"\"Calculate derived attributes after initialization.\"\"\"
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

        # Generate a hash based on relevant config options for versioning
        # Build the dictionary for hashing manually to avoid issues with asdict() and init=False fields
        hash_relevant_config = {}
        exclude_from_hash = {
            'root', 'version_name', 'qm9_url', 'raw_archive_name', 'raw_sdf_name',
            'force_download', 'force_process', 'num_workers', 'split_definition_path', 'load_split',
            '_config_hash', '_atom_symbol_map', '_bond_type_map', '_actual_target_keys' # Also exclude internal calculated fields
        }
        for field_name, field_obj in self.__dataclass_fields__.items():
            if field_name not in exclude_from_hash and field_obj.init: # Only include initialized fields relevant to data content
                hash_relevant_config[field_name] = getattr(self, field_name)

        # Ensure consistent serialization for hashing (sort keys)
        config_str = json.dumps(hash_relevant_config, sort_keys=True, default=str)
        self._config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8] # Short hash
        logging.info(f"Configuration hash for versioning: {self._config_hash}")

        # Validate split options
        if self.load_split and not self.split_definition_path:
            raise ValueError("`load_split` requires `split_definition_path` to be set.")
        if self.load_split and self.load_split not in ['train', 'val', 'test']:
            raise ValueError("`load_split` must be one of 'train', 'val', 'test' or None.")

# --- Helper Function: Molecule to PyG Data (now using Config) ---

def rdkit_mol_to_pyg_data_configured(mol: Chem.Mol, config: QM9Config) -> Optional[Data]:
    """
    Converts an RDKit molecule to PyG Data using settings from QM9Config.
    Handles optional feature extraction. Uses canonical SMILES.
    """
    canonical_smiles = "[Unavailable]"
    try:
        # --- Basic Molecule Checks & Canonical SMILES ---
        if mol is None: return None
        num_atoms_initial = mol.GetNumAtoms()
        if num_atoms_initial == 0: return None

        try:
            # Generate canonical SMILES for reliable identification and duplicate checking
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            canonical_smiles = "[Canonical SMILES Generation Failed]"

        # --- Optional: Remove Hydrogens (Handled by SDMolSupplier setting in QM9EnhancedDataset) ---
        # We assume the input `mol` object already has hydrogens removed or kept based on config.

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            logging.warning(f"Molecule {canonical_smiles} has 0 atoms after potential processing. Skipping.")
            return None


        # --- Get Conformer Positions ---
        try:
            conformer = mol.GetConformer()
            positions = conformer.GetPositions()
            pos_tensor = torch.tensor(positions, dtype=torch.float)
            if pos_tensor.shape[0] != num_atoms:
                 logging.warning(f"Skipping {canonical_smiles}: Mismatch between num_atoms ({num_atoms}) and conformer positions ({pos_tensor.shape[0]}).")
                 return None
        except (AttributeError, ValueError) as e:
            logging.warning(f"Skipping {canonical_smiles}: Error getting conformer/positions: {e}")
            return None

        # --- Atom Features ---
        atom_features_list = []
        atomic_numbers = []
        for i, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in config._atom_symbol_map:
                logging.warning(f"Skipping {canonical_smiles}: Unsupported atom type '{atom_symbol}'. Allowed: {config.allowed_atom_symbols}")
                return None

            # Basic one-hot encoding of atom type
            atom_type_onehot = torch.zeros(len(config.allowed_atom_symbols), dtype=torch.float)
            atom_type_onehot[config._atom_symbol_map[atom_symbol]] = 1.0
            features = [atom_type_onehot]
            atomic_numbers.append(atom.GetAtomicNum())

            # Optional features
            if config.add_atomic_mass:
                # Using raw atomic mass, removed arbitrary scaling
                features.append(torch.tensor([atom.GetMass()], dtype=torch.float))
            if config.add_formal_charge:
                features.append(torch.tensor([atom.GetFormalCharge()], dtype=torch.float))
            if config.add_hybridization:
                hyb = atom.GetHybridization()
                # Common hybridization types
                hyb_map = {Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1, Chem.HybridizationType.SP3: 2, Chem.HybridizationType.SP3D: 3, Chem.HybridizationType.SP3D2: 4}
                hyb_onehot = torch.zeros(5, dtype=torch.float)
                if hyb in hyb_map: hyb_onehot[hyb_map[hyb]] = 1.0
                # TODO: Consider adding a 6th category for 'Other/Unspecified' hybridization if needed.
                features.append(hyb_onehot)
            if config.add_is_aromatic_atom:
                features.append(torch.tensor([1.0 if atom.GetIsAromatic() else 0.0], dtype=torch.float))
            if config.add_is_in_ring_atom:
                 features.append(torch.tensor([1.0 if atom.IsInRing() else 0.0], dtype=torch.float))

            atom_features_list.append(torch.cat(features))

        x_tensor = torch.stack(atom_features_list, dim=0)

        # --- Edge Features & Indices ---
        edge_indices_list = []
        edge_features_list = []
        if num_atoms > 1: # Only process bonds if there's more than one atom
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                if bond_type not in config._bond_type_map:
                    logging.warning(f"Skipping {canonical_smiles}: Unsupported bond type '{bond_type}'. Allowed: {config.allowed_bond_types}")
                    return None

                start_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()

                # Basic one-hot encoding of bond type
                bond_type_onehot = torch.zeros(len(config.allowed_bond_types), dtype=torch.float)
                bond_type_onehot[config._bond_type_map[bond_type]] = 1.0
                features = [bond_type_onehot]

                # Optional features
                if config.add_is_conjugated_bond:
                    features.append(torch.tensor([1.0 if bond.GetIsConjugated() else 0.0], dtype=torch.float))
                if config.add_is_in_ring_bond:
                    features.append(torch.tensor([1.0 if bond.IsInRing() else 0.0], dtype=torch.float))

                edge_feature_vector = torch.cat(features)

                # Add edges in both directions
                edge_indices_list.append((start_atom_idx, end_atom_idx))
                edge_indices_list.append((end_atom_idx, start_atom_idx))
                edge_features_list.append(edge_feature_vector)
                edge_features_list.append(edge_feature_vector) # Same features for reverse direction

        # Handle cases with 0 or 1 atom (no bonds)
        num_edges = len(edge_indices_list)
        if num_edges > 0:
            edge_index_tensor = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.stack(edge_features_list, dim=0)
            # Infer edge feature dimension from first element
            edge_feature_dim = edge_features_list[0].shape[0]
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            # Try to infer edge feature dimension from atom features if possible, else use 0
            # This requires knowing how many edge features we intended to create
            # Let's calculate expected edge feature dim based on config:
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
                logging.warning(f"Skipping {canonical_smiles}: Target property '{key}' not found.")
                return None
            except ValueError:
                logging.warning(f"Skipping {canonical_smiles}: Could not convert property '{key}' value '{mol_props.get(key)}' to float.")
                return None

        y_tensor = torch.tensor(target_values, dtype=torch.float).unsqueeze(0) # Shape [1, num_targets]

        # --- Create PyG Data Object ---
        data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            pos=pos_tensor,
            y=y_tensor,
            smiles=canonical_smiles, # Store canonical SMILES
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            num_nodes=num_atoms
        )

        # --- Final Validation ---
        validation_result = data.validate(raise_on_error=False)
        if not validation_result:
             # Log specific validation errors if the attribute exists
             errors = getattr(data, 'validation_errors', 'N/A')
             logging.warning(f"Data validation failed for {canonical_smiles}. Skipping. Errors: {errors}")
             return None

        return data

    except Exception as e:
        logging.error(f"Unexpected error processing molecule (SMILES: {canonical_smiles}): {e}")
        # Log traceback for unexpected errors, especially during development/debugging
        logging.debug(traceback.format_exc())
        return None


# --- Worker function for multiprocessing ---
# Needs to be defined at the top level to be pickleable

_GLOBAL_CONFIG = None # Global variable to hold config for workers
_GLOBAL_SDF_PATH = None # Global variable to hold SDF path for workers

def init_worker(config_dict: dict, sdf_path: str):
    """Initializer for multiprocessing worker pool."""
    global _GLOBAL_CONFIG, _GLOBAL_SDF_PATH
    try:
        _GLOBAL_CONFIG = QM9Config(**config_dict)
        _GLOBAL_SDF_PATH = sdf_path
        # Disable RDKit logging again within the worker process
        RDLogger.DisableLog('rdApp.*')
        # Optional: Log worker initialization
        # logging.debug(f"Worker {current_process().name} initialized.")
    except Exception as e:
        logging.error(f"Error initializing worker {current_process().name}: {e}")
        _GLOBAL_CONFIG = None # Ensure worker is marked as uninitialized on error
        _GLOBAL_SDF_PATH = None


def process_mol_worker(mol_index: int) -> Optional[Data]:
    """Processes a single molecule by index in a worker process."""
    global _GLOBAL_CONFIG, _GLOBAL_SDF_PATH
    if _GLOBAL_CONFIG is None or _GLOBAL_SDF_PATH is None:
        # Initializer likely failed for this worker
        logging.error(f"Worker {current_process().name} not initialized properly. Skipping task for index {mol_index}.")
        return None

    # Each worker creates its own supplier instance.
    # Seeking by index in SDMolSupplier is generally efficient.
    try:
        # Crucially set removeHs based on config BEFORE accessing the molecule
        supplier = Chem.SDMolSupplier(_GLOBAL_SDF_PATH, removeHs=(not _GLOBAL_CONFIG.use_h_atoms), sanitize=True)
        if mol_index < 0 or mol_index >= len(supplier):
             logging.error(f"Worker {current_process().name}: Invalid index {mol_index} requested (max: {len(supplier)-1}).")
             return None
        mol = supplier[mol_index] # Access molecule by index
        # Explicitly delete supplier to potentially release file handle sooner, though Python's GC should handle it.
        del supplier

    except IndexError:
        # This case might be covered by the check above, but kept for robustness.
        logging.error(f"Worker {current_process().name}: Index {mol_index} out of bounds for supplier.")
        return None
    except Exception as e:
        # Catch potential errors during supplier creation or molecule access
        logging.error(f"Worker {current_process().name}: Error creating supplier or accessing index {mol_index}: {e}")
        return None # Cannot proceed if supplier fails

    if mol is None:
        # SDMolSupplier returns None if parsing that specific record fails
        # Logging this can be very verbose if the SDF has many bad entries.
        # logging.debug(f"Worker {current_process().name}: RDKit failed to parse molecule at index {mol_index}. Skipping.")
        return None

    # Process the molecule using the main helper function and the globally stored config
    return rdkit_mol_to_pyg_data_configured(mol, _GLOBAL_CONFIG)


# --- Enhanced QM9 Dataset Class ---

class QM9EnhancedDataset(InMemoryDataset):
    """
    Enhanced QM9 Dataset Loader with configurable features, parallel processing,
    versioning based on config, optional splits, and duplicate checking.
    Uses canonical SMILES for duplicate checks. Includes refinements based on feedback.
    """
    def __init__(self,
                 config: QM9Config,
                 transform: Optional[Callable[[Data], Data]] = None,
                 pre_transform: Optional[Callable[[Data], Data]] = None,
                 pre_filter: Optional[Callable[[Data], bool]] = None):

        self.config = config
        self._raw_sdf_path = osp.join(self.config.root, 'raw', self.config.raw_sdf_name)
        # Processed directory now includes the manual version name
        self._processed_dir_path = osp.join(self.config.root, 'processed', self.config.version_name)

        logging.info(f"Initializing QM9EnhancedDataset in root: {self.config.root}")
        logging.info(f"Processing version name: {self.config.version_name}")
        logging.info(f"Config hash (determines processed filename): {self.config._config_hash}")
        logging.info(f"Number of workers for processing: {self.config.num_workers}")
        if self.config.load_split:
             logging.info(f"Requesting data split: {self.config.load_split} using definition: {self.config.split_definition_path}")

        # --- Important: Call super().__init__ AFTER setting up paths ---
        # super().__init__ calls self.processed_dir, self.raw_dir etc.
        super().__init__(self.config.root, transform, pre_transform, pre_filter)

        # Load data - super().__init__ ensures it's processed if needed.
        try:
             # self.processed_paths is defined by the superclass based on processed_dir and processed_file_names
             self.data, self.slices = torch.load(self.processed_paths[0])
             logging.info(f"Dataset loaded successfully from {self.processed_paths[0]}")
        except FileNotFoundError:
             logging.error(f"Processed file not found at {self.processed_paths[0]} after initialization.")
             logging.error("Ensure processing completed successfully or check paths/config hash.")
             raise # Re-raise the error as dataset loading failed
        except Exception as e:
            logging.error(f"Error loading processed file {self.processed_paths[0]}: {e}")
            raise

        # --- Apply Splits (if requested) ---
        # This modifies self.data and self.slices in place for the current instance
        if self.config.load_split:
            self._apply_split()

    # --- Override path properties to use config ---
    @property
    def raw_dir(self) -> str:
        return osp.join(self.config.root, 'raw')

    @property
    def processed_dir(self) -> str:
        # The directory structure is root/processed/version_name/
        os.makedirs(self._processed_dir_path, exist_ok=True)
        return self._processed_dir_path

    @property
    def raw_file_names(self) -> List[str]:
        return [self.config.raw_sdf_name]

    @property
    def processed_file_names(self) -> List[str]:
        # Filename includes the config hash for automatic versioning based on content generation logic
        return [f'qm9_pyg_{self.config._config_hash}.pt']

    def download(self):
        """Downloads and extracts the QM9 SDF file if not present or forced."""
        # Ensure raw directory exists before download attempt
        os.makedirs(self.raw_dir, exist_ok=True)
        raw_archive_path = osp.join(self.raw_dir, self.config.raw_archive_name)

        if osp.exists(self._raw_sdf_path) and not self.config.force_download:
            logging.info(f"Raw file {self.config.raw_sdf_name} already exists at {self._raw_sdf_path}. Skipping download.")
            return

        if osp.exists(self._raw_sdf_path) and self.config.force_download:
             logging.info(f"Force download enabled. Removing existing raw file: {self._raw_sdf_path}")
             try:
                 os.remove(self._raw_sdf_path)
             except OSError as e:
                 logging.warning(f"Could not remove existing raw file {self._raw_sdf_path}: {e}")


        logging.info(f"Downloading {self.config.qm9_url} to {raw_archive_path}...")
        try:
            response = requests.get(self.config.qm9_url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {self.config.raw_archive_name}", leave=False) as pbar:
                with open(raw_archive_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            if total_size != 0 and pbar.n != total_size:
                raise IOError(f"Download incomplete: got {pbar.n} of {total_size} bytes.")
            logging.info("Download complete.")

            logging.info(f"Extracting {self.config.raw_sdf_name} from {raw_archive_path}...")
            with tarfile.open(raw_archive_path, 'r:gz') as tar:
                 member_info = None
                 for member in tar.getmembers():
                      if member.name.endswith(self.config.raw_sdf_name):
                          member_info = member
                          logging.info(f"Found SDF file '{member.name}' in archive.")
                          break
                 if member_info is None: raise FileNotFoundError(f"SDF file '{self.config.raw_sdf_name}' not found in archive.")
                 # Extract to raw_dir root, ensuring correct final path
                 member_info.name = osp.basename(member_info.name)
                 tar.extract(member_info, path=self.raw_dir)
                 # Verify it's at the expected location
                 if not osp.exists(self._raw_sdf_path):
                     raise FileNotFoundError(f"Extracted file not found at expected path: {self._raw_sdf_path}")
            logging.info(f"Extraction complete to {self._raw_sdf_path}.")

            # Clean up archive
            try:
                os.remove(raw_archive_path)
                logging.info(f"Removed archive {raw_archive_path}.")
            except OSError as e:
                 logging.warning(f"Could not remove archive {raw_archive_path}: {e}")

        except (requests.exceptions.RequestException, tarfile.TarError, FileNotFoundError, IOError, OSError) as e:
            logging.error(f"Download/Extraction failed: {e}")
            # Clean up potentially corrupted files
            if osp.exists(raw_archive_path):
                try: os.remove(raw_archive_path)
                except OSError: pass
            if osp.exists(self._raw_sdf_path):
                 try: os.remove(self._raw_sdf_path)
                 except OSError: pass
            raise RuntimeError("Failed to download or extract QM9 data.") from e

    def process(self):
        """Processes the raw SDF file into PyG Data objects using multiprocessing."""
        logging.info(f"Starting parallel processing using up to {self.config.num_workers} workers...")
        start_time = time.time()

        # --- Determine number of molecules ---
        logging.info(f"Counting molecules in {self._raw_sdf_path}...")
        try:
             # Ensure consistency: use the same removeHs setting for counting
             counter_supplier = Chem.SDMolSupplier(self._raw_sdf_path, removeHs=(not self.config.use_h_atoms), sanitize=True)
             # Iterating is safer than len() for some RDKit versions/large files
             num_molecules = 0
             for _ in counter_supplier:
                 num_molecules += 1
             del counter_supplier
             logging.info(f"Counted {num_molecules} molecule entries.")
             if num_molecules == 0:
                 raise ValueError("SDF file contains no processable molecule entries.")
        except Exception as e:
            logging.error(f"Failed to read or count molecules in {self._raw_sdf_path}: {e}")
            raise

        # --- Setup Multiprocessing Pool ---
        # Pass config as dict and SDF path to initializer
        init_args = (asdict(self.config), self._raw_sdf_path)
        data_list = []
        skipped_count = 0
        processed_count = 0
        seen_smiles: Set[str] = set()
        duplicate_count = 0

        # Use try-with-resources for the Pool for automatic cleanup
        try:
            # Choose number of workers, but not more than molecules available
            actual_workers = min(self.config.num_workers, num_molecules)
            if actual_workers < self.config.num_workers:
                logging.info(f"Using {actual_workers} workers (limited by molecule count).")

            # Create the pool
            with Pool(processes=actual_workers, initializer=init_worker, initargs=init_args) as pool:
                # Use imap_unordered for potential memory efficiency and faster feedback
                results_iterator = pool.imap_unordered(process_mol_worker, range(num_molecules))

                # Process results as they become available
                with tqdm(total=num_molecules, desc="Processing Molecules (Parallel)", smoothing=0.1) as pbar:
                    for i, result_data in enumerate(results_iterator):
                        pbar.update(1)
                        if result_data is not None:
                            # --- Duplicate Check (using canonical SMILES) ---
                            if self.config.check_duplicates:
                                if result_data.smiles in seen_smiles:
                                    # Log first few duplicates, then suppress to avoid flooding
                                    if duplicate_count < 5:
                                         logging.debug(f"Duplicate canonical SMILES detected and skipped: {result_data.smiles}")
                                    elif duplicate_count == 5:
                                         logging.debug("Further duplicate messages suppressed...")
                                    duplicate_count += 1
                                    skipped_count += 1
                                    continue # Skip duplicate
                                else:
                                    # Only add canonical SMILES if check_duplicates is enabled
                                    seen_smiles.add(result_data.smiles)

                            # --- Apply pre_filter/pre_transform (in main process) ---
                            # These operate on the successfully processed Data object
                            data_valid = True
                            if self.pre_filter is not None and not self.pre_filter(result_data):
                                skipped_count += 1
                                data_valid = False
                            if data_valid and self.pre_transform is not None:
                                try:
                                    transformed_data = self.pre_transform(result_data)
                                    if transformed_data is None: # Handle transforms that might filter data
                                        logging.warning(f"Molecule {getattr(result_data, 'smiles', '[SMILES unavailable]')} became None after pre_transform. Skipping.")
                                        skipped_count += 1
                                        data_valid = False
                                    else:
                                        result_data = transformed_data # Use the transformed data
                                except Exception as e:
                                    logging.error(f"Error during pre_transform for molecule {getattr(result_data, 'smiles', '[SMILES unavailable]')}: {e}")
                                    skipped_count += 1
                                    data_valid = False

                            # Add to list if valid and passed filters/transforms
                            if data_valid:
                                data_list.append(result_data)
                                processed_count += 1
                        else:
                            # Worker returned None (error logged within worker/helper)
                            skipped_count += 1
                        # Optional: Log progress periodically
                        # if (i + 1) % 10000 == 0:
                        #    logging.debug(f"Processed {i+1}/{num_molecules} entries...")

        except Exception as e:
             logging.error(f"An error occurred during multiprocessing: {e}")
             logging.error(traceback.format_exc())
             raise RuntimeError("Multiprocessing pool encountered an error.") from e
        finally:
            # Ensure the pool is properly terminated even if tqdm raises an exception
             pass # 'with Pool(...)' handles closing and joining

        end_time = time.time()
        processing_duration = end_time - start_time
        logging.info(f"Parallel processing finished in {processing_duration:.2f} seconds.")
        if processing_duration > 0:
             mols_per_sec = processed_count / processing_duration
             logging.info(f"Processing speed: {mols_per_sec:.2f} molecules/second (using {actual_workers} workers).")


        if duplicate_count > 0:
             logging.info(f"Detected and skipped {duplicate_count} duplicate molecules based on canonical SMILES.")
        if skipped_count > processed_count: # Includes duplicates and other skips
            logging.warning(f"Total skipped/filtered entries: {skipped_count}. Check logs for details.")
        if not data_list:
             logging.error("No molecules were processed successfully after filtering and duplicate removal.")
             # Provide more context if possible
             logging.error(f"Initial count: {num_molecules}, Processed: {processed_count}, Skipped: {skipped_count}, Duplicates: {duplicate_count}")
             raise ValueError("Processing resulted in an empty dataset.")

        logging.info(f"Successfully processed and retained {processed_count} unique molecules.")
        logging.info(f"Collating {len(data_list)} data objects...")
        data, slices = self.collate(data_list)

        processed_path = self.processed_paths[0] # Get the target path based on hash
        logging.info(f"Saving processed data to {processed_path}...")
        try:
            torch.save((data, slices), processed_path)
            logging.info("Saving complete.")
        except Exception as e:
             logging.error(f"Failed to save processed data to {processed_path}: {e}")
             # Attempt to remove potentially corrupted file
             if osp.exists(processed_path):
                 try: os.remove(processed_path)
                 except OSError: pass
             raise


    def _apply_split(self):
        """Filters the loaded data based on the requested split definition file."""
        if not self.config.load_split:
            # Should not happen if called correctly, but acts as a safeguard
            logging.debug("No split requested in config, skipping split application.")
            return
        if not self.config.split_definition_path:
             logging.error("load_split is set, but split_definition_path is missing.")
             raise ValueError("Split definition path required when load_split is specified.")

        split_file = Path(self.config.split_definition_path)
        if not split_file.is_file():
            logging.error(f"Split definition file not found at: {split_file}")
            raise FileNotFoundError(f"Split file not found: {split_file}")

        try:
            logging.info(f"Loading split indices from {split_file}...")
            split_dict = torch.load(split_file)
            logging.info(f"Available splits in file: {list(split_dict.keys())}")

            split_key = f"{self.config.load_split}_idx" # e.g., 'train_idx', 'val_idx', 'test_idx'
            if split_key not in split_dict:
                 raise ValueError(f"Split key '{split_key}' not found in {split_file}. Available keys: {list(split_dict.keys())}")

            indices = split_dict[split_key]
            # Ensure indices are a LongTensor
            if not isinstance(indices, torch.Tensor):
                 indices = torch.tensor(indices, dtype=torch.long)
            if indices.dtype != torch.long:
                 indices = indices.to(torch.long)

            num_total_before_split = self.len()
            num_indices = len(indices)
            logging.info(f"Applying filter for split '{self.config.load_split}' using {num_indices} indices (from {num_total_before_split} total).")

            # Validate indices bounds (optional but good practice)
            if num_indices > 0:
                 max_idx = indices.max().item()
                 if max_idx >= num_total_before_split:
                     logging.warning(f"Maximum index in split file ({max_idx}) is out of bounds for loaded data size ({num_total_before_split}). Clamping or check split file.")
                     # Option 1: Clamp indices (might hide issues)
                     # indices = indices[indices < num_total_before_split]
                     # Option 2: Raise error (safer)
                     raise IndexError(f"Split index {max_idx} out of bounds for dataset size {num_total_before_split}.")


            # Use PyG's index_select to get the subset of data
            # This replaces self.data and self.slices with the selected subset
            self.data, self.slices = self.index_select(indices)

            logging.info(f"Dataset size after applying split '{self.config.load_split}': {self.len()}")
            if self.len() != num_indices and num_indices > 0: # Check if length matches expected indices (after clamping/filtering)
                 logging.warning(f"Final dataset size {self.len()} does not match number of valid indices {num_indices}. Possible issues during index_select or with indices.")

        except Exception as e:
             logging.error(f"Failed to load or apply split from {split_file}: {e}")
             raise RuntimeError("Error during data splitting.") from e


# --- Example Usage Script ---
if __name__ == '__main__':
    logging.info("--- Starting Enhanced QM9 Dataset Preparation Script ---")

    # --- Configuration ---
    # Adjust configuration as needed
    config = QM9Config(
        root=osp.join(os.getcwd(), 'data', 'QM9_Refined'),
        version_name="v1_canon_nodup",
        target_keys_to_load=['U0', 'gap', 'mu'], # Example targets
        add_formal_charge=True,
        add_is_aromatic_atom=True,
        add_hybridization=True,
        check_duplicates=True, # Enable duplicate check using canonical SMILES
        num_workers=max(1, cpu_count() - 2), # Leave some cores free
        # --- Split Configuration ---
        split_definition_path=osp.join(os.getcwd(), 'data', 'qm9_std_splits.pt'), # Use a standard split file
        load_split='train' # Load only the training split
        # force_process=True # Uncomment to force reprocessing
    )

    # --- Create Dummy Split File (if necessary for testing) ---
    # !! Replace this with downloading/generating actual standard splits !!
    dummy_split_path = Path(config.split_definition_path) if config.split_definition_path else None
    if dummy_split_path and not dummy_split_path.exists():
         logging.warning(f"Creating DUMMY standard split file at: {dummy_split_path}")
         logging.warning("!!! THIS IS FOR DEMO/TESTING ONLY - USE REAL QM9 SPLITS !!!")
         os.makedirs(dummy_split_path.parent, exist_ok=True)
         # Approximate sizes based on common QM9 splits (e.g., SchNet)
         num_total_qm9_approx = 133885 # Often cited number after filtering initial set
         num_val = 10000
         num_test = 10000 # Or sometimes rest after train/val split from ~110k
         num_train = num_total_qm9_approx - num_val - num_test

         indices = torch.randperm(num_total_qm9_approx)
         train_idx = indices[:num_train]
         val_idx = indices[num_train:num_train + num_val]
         # Ensure test_idx covers the rest, handle off-by-one if needed
         test_idx = indices[num_train + num_val:]
         if len(train_idx) + len(val_idx) + len(test_idx) != num_total_qm9_approx:
              logging.warning("Dummy split sizes don't perfectly match total approximation.")

         torch.save({'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}, dummy_split_path)
         logging.info(f"Dummy split file created with sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    elif not dummy_split_path:
         logging.info("No split_definition_path provided in config, skipping split file check/creation.")
    else:
         logging.info(f"Using existing split file found at: {dummy_split_path}")
    # --- End of Dummy Split File Creation ---


    try:
        # Ensure root directory exists
        os.makedirs(config.root, exist_ok=True)

        logging.info("Instantiating QM9EnhancedDataset with configuration...")
        start_instantiation = time.time()
        qm9_dataset = QM9EnhancedDataset(config=config)
        end_instantiation = time.time()
        logging.info(f"Dataset instantiation took {end_instantiation - start_instantiation:.2f} seconds.")

        # --- Basic Dataset Inspection ---
        logging.info("\n--- Dataset Inspection ---")
        logging.info(f"Dataset object: {qm9_dataset}")
        # logging.info(f"Config used: {qm9_dataset.config}") # Can be verbose
        logging.info(f"Number of graphs in loaded split ('{config.load_split}'): {len(qm9_dataset)}")

        if len(qm9_dataset) > 0:
            first_data = qm9_dataset[0]
            logging.info("\n--- Inspecting First Data Point ---")
            logging.info(f"Data object:\n{first_data}")
            logging.info(f"Node feature shape (x): {first_data.x.shape}")
            logging.info(f"Edge feature shape (edge_attr): {first_data.edge_attr.shape}")
            logging.info(f"Target properties shape (y): {first_data.y.shape}")
            logging.info(f"Expected targets loaded: {config._actual_target_keys}")
            logging.info(f"Target values (y): {first_data.y}")
            logging.info(f"Canonical SMILES: {first_data.smiles}")

            # Verification check
            assert first_data.y.shape[1] == len(config._actual_target_keys), "Mismatch in number of target properties"
            logging.info("Target count matches configuration.")

            # Check feature dimensions (example)
            expected_node_dim = len(config.allowed_atom_symbols)
            if config.add_atomic_mass: expected_node_dim += 1
            if config.add_formal_charge: expected_node_dim += 1
            if config.add_hybridization: expected_node_dim += 5 # Size of one-hot
            if config.add_is_aromatic_atom: expected_node_dim += 1
            if config.add_is_in_ring_atom: expected_node_dim += 1
            assert first_data.x.shape[1] == expected_node_dim, f"Mismatch in node feature dimension: expected {expected_node_dim}, got {first_data.x.shape[1]}"
            logging.info(f"Node feature dimension ({first_data.x.shape[1]}) matches configuration.")

            expected_edge_dim = len(config.allowed_bond_types)
            if config.add_is_conjugated_bond: expected_edge_dim += 1
            if config.add_is_in_ring_bond: expected_edge_dim += 1
            if first_data.num_edges > 0:
                 assert first_data.edge_attr.shape[1] == expected_edge_dim, f"Mismatch in edge feature dimension: expected {expected_edge_dim}, got {first_data.edge_attr.shape[1]}"
                 logging.info(f"Edge feature dimension ({first_data.edge_attr.shape[1]}) matches configuration.")
            else:
                 # Check shape[1] even if shape[0] is 0
                 assert first_data.edge_attr.shape[1] == expected_edge_dim, f"Mismatch in edge feature dimension (0 edges): expected {expected_edge_dim}, got {first_data.edge_attr.shape[1]}"
                 logging.info(f"Edge feature dimension ({first_data.edge_attr.shape[1]}) matches configuration (for 0 edges).")


        else:
            logging.warning(f"Dataset split '{config.load_split}' is empty after processing/filtering.")

        logging.info("\n--- Enhanced QM9 Dataset script finished successfully! ---")

    except (FileNotFoundError, ValueError, RuntimeError, ConnectionError, AssertionError, IndexError) as e:
        logging.error(f"\n--- A critical error occurred during dataset preparation or verification ---")
        logging.error(f"Error Type: {type(e).__name__}")
        logging.error(f"Error Details: {e}")
        logging.exception("Traceback:")
    except Exception as e:
        logging.error(f"\n--- An unexpected error occurred ---")
        logging.error(f"Error Type: {type(e).__name__}")
        logging.error(f"Error Details: {e}")
        logging.exception("Traceback:")
