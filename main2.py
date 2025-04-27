# -*- coding: utf-8 -*-
"""
Erweitertes Modul zur Verarbeitung des QM9-Datensatzes in PyTorch Geometric Data-Objekte.

Dieses Skript implementiert einen robusten Workflow, der Folgendes umfasst:
- Laden der Konfiguration aus einer YAML-Datei.
- Parallele Verarbeitung von Molekülen zur Leistungssteigerung.
- Detaillierte Fehlererfassung für fehlgeschlagene Moleküle.
- Optionale Standardisierung von Zielwerten mit Speicherung der Parameter.
- Download und Extraktion der Rohdaten.
- Erstellung von PyG InMemoryDataset mit versionierten Ausgaben basierend auf der Konfiguration.
- Speicherung umfangreicher Metadaten.
- Kommandozeilen-Interface zur Steuerung.
"""

# ==============================================================================
# Block 1: Import von Abhängigkeiten
# ==============================================================================
import os
import os.path as osp
import tarfile
import requests
import shutil
import logging
import traceback
import hashlib
import json
import time
import argparse # Für Kommandozeilenargumente
from typing import List, Dict, Tuple, Optional, Callable, Any, Set, Union, NamedTuple
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from collections import defaultdict
import math # Für NaN-Checks

# PyTorch und PyG Imports
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.typing import Tensor

# RDKit Imports
try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit import RDLogger
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    BT = type('BondType', (), {'SINGLE': None, 'DOUBLE': None, 'TRIPLE': None, 'AROMATIC': None})
    # Fehler wird später in der Konfigprüfung ausgelöst

# TQDM für Fortschrittsbalken
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        logging.info("tqdm nicht verfügbar. Fortschrittsbalken werden nicht angezeigt.")
        return iterable

# Multiprocessing
# Verwende concurrent.futures für modernere API als multiprocessing.Pool direkt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp # Immer noch nützlich für cpu_count etc.

# YAML für Konfiguration
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # Fehler wird später ausgelöst, wenn Konfig geladen werden soll


# ==============================================================================
# Block 2: Konfiguration der Protokollierung (Logging)
# ==============================================================================
# Logging-Setup: Wird jetzt in der main-Funktion konfiguriert, um das Level steuerbar zu machen

# Globale Zähler für limitierte Warnungen (problematisch bei Parallelisierung, daher nicht mehr genutzt)
# Stattdessen wird auf detaillierte Fehlererfassung gesetzt.

# Hilfsfunktion für Logging (jetzt einfacher, da keine Limitierung mehr)
def log_warning(message: str):
    logging.warning(message)

def log_error(message: str, exc_info: bool = False):
    logging.error(message, exc_info=exc_info)

# Deaktiviere RDKit Logger frühzeitig
if RDKIT_AVAILABLE:
    RDLogger.DisableLog('rdApp.*')
    # Info-Meldung kommt nach Logging-Setup in main()

# ==============================================================================
# Block 3: Konstanten und Mappings
# ==============================================================================
BOND_TYPE_TO_NAME: Dict[BT, str] = {
    BT.SINGLE: 'SINGLE', BT.DOUBLE: 'DOUBLE', BT.TRIPLE: 'TRIPLE', BT.AROMATIC: 'AROMATIC'
}
NAME_TO_BOND_TYPE: Dict[str, BT] = {v: k for k, v in BOND_TYPE_TO_NAME.items()}

# Ergebnis-Typen für Worker
class ProcessingResult(NamedTuple):
    success: bool
    identifier: Union[int, str] # Index oder SMILES
    payload: Union[Data, Tuple[str, str]] # Data-Objekt oder (Fehlertyp, Fehlermeldung)

# ==============================================================================
# Block 4: Konfigurations-Datenklasse (QM9Config)
# ==============================================================================
@dataclass
class TargetScalingConfig:
    """Konfiguration für die Skalierung von Zielwerten."""
    method: Optional[str] = None  # z.B. 'standardize', 'normalize' (MinMax), 'log1p', None
    # Parameter werden zur Laufzeit berechnet und hier *temporär* oder in Metadaten gespeichert
    params: Dict[str, Dict[str, float]] = field(default_factory=dict) # z.B. {'mu': {'mean': X, 'std': Y}, ...}

@dataclass
class QM9Config:
    """
    Konfigurations-Datenklasse für die Verarbeitung des QM9-Datensatzes.
    Wird typischerweise aus einer YAML-Datei geladen.
    """
    # --- Pfade und Versionierung ---
    root_dir: str = 'data/QM9_Processed' # Basisverzeichnis für alle Ausgaben
    version_name: str = "default_v1"     # Unterordner für diese spezifische Verarbeitung

    # --- Molekülverarbeitung ---
    use_h_atoms: bool = True
    allowed_atom_symbols: List[str] = field(default_factory=lambda: ['H', 'C', 'N', 'O', 'F'])
    allowed_bond_type_names: List[str] = field(default_factory=lambda: ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])

    # --- Optionale Features ---
    add_atomic_mass: bool = False
    add_formal_charge: bool = False
    add_hybridization: bool = False
    add_is_aromatic_atom: bool = False
    add_is_in_ring_atom: bool = False
    add_is_conjugated_bond: bool = False
    add_is_in_ring_bond: bool = False

    # --- Zielvariablen (Targets) ---
    all_available_targets: List[str] = field(default_factory=lambda: [
        'mu', 'alpha', 'HOMO', 'LUMO', 'gap', 'R2', 'ZPVE', 'U0', 'U', 'H', 'G', 'CV'
    ])
    target_keys_to_load: Optional[List[str]] = None # None = lade alle 'all_available_targets'

    # --- Zielwert-Skalierung ---
    target_scaling: TargetScalingConfig = field(default_factory=TargetScalingConfig)

    # --- Download ---
    qm9_url: str = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
    raw_archive_name: str = 'gdb9.tar.gz'
    raw_sdf_name: str = 'gdb9.sdf'

    # --- Verarbeitung ---
    num_workers: Optional[int] = None # None = cpu_count(), 1 = sequentiell, >1 = parallel
    check_duplicates: bool = True     # SMILES-basierter Duplikat-Check
    # Hinweis: Robustere Checks (Graph-Isomorphismus) sind hier nicht implementiert.
    processing_chunksize: int = 1000  # Wie viele Moleküle pro Worker-Task (für imap)

    # --- Aufteilung (Splitting) ---
    # Wird jetzt nach der Initialisierung angewendet, nicht Teil der Hash-Konfiguration
    split_definition_path: Optional[str] = None
    load_split: Optional[str] = None

    # --- Interne / Abgeleitete Attribute ---
    _actual_target_keys: List[str] = field(init=False, repr=False)
    _atom_symbol_map: Dict[str, int] = field(init=False, repr=False)
    _allowed_bond_types_internal: List[BT] = field(init=False, repr=False)
    _bond_type_map: Dict[BT, int] = field(init=False, repr=False)
    _config_hash: str = field(init=False, repr=False)

    def generate_config_hash(self, config_dict_for_hash: Dict) -> str:
        """Generiert einen Hash aus einem Konfigurations-Dictionary."""
        # Sortiere das Dictionary, um einen stabilen Hash zu gewährleisten
        # Konvertiere komplexe Objekte wie TargetScalingConfig in ein serialisierbares Format
        def default_serializer(obj):
            if isinstance(obj, TargetScalingConfig):
                # Nur die Methode ist relevant für den Hash, nicht die berechneten Parameter
                return {'method': obj.method}
            return str(obj) # Fallback

        config_string = json.dumps(config_dict_for_hash, sort_keys=True, default=default_serializer)
        return hashlib.md5(config_string.encode('utf-8')).hexdigest()[:10]

    def __post_init__(self):
        """Validiert Konfiguration und berechnet abgeleitete Attribute."""
        logging.debug("Starte QM9Config Validierung und Initialisierung...")

        # --- RDKit Check ---
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit ist erforderlich, konnte aber nicht importiert werden.")

        # --- Bindungstypen ---
        invalid_bonds = [n for n in self.allowed_bond_type_names if n not in NAME_TO_BOND_TYPE]
        if invalid_bonds:
            raise ValueError(f"Ungültige Bindungstypen: {invalid_bonds}")
        self._allowed_bond_types_internal = [NAME_TO_BOND_TYPE[n] for n in self.allowed_bond_type_names]
        self._bond_type_map = {bt: i for i, bt in enumerate(self._allowed_bond_types_internal)}

        # --- Atomsymbole ---
        self._atom_symbol_map = {s: i for i, s in enumerate(self.allowed_atom_symbols)}

        # --- Zielwerte ---
        prop_set = set(self.all_available_targets)
        if self.target_keys_to_load is None:
            self._actual_target_keys = list(self.all_available_targets)
        else:
            invalid_keys = [k for k in self.target_keys_to_load if k not in prop_set]
            if invalid_keys:
                raise ValueError(f"Ungültige target_keys_to_load: {invalid_keys}")
            self._actual_target_keys = list(self.target_keys_to_load)
        logging.info(f"Zielwerte (Targets) zu laden: {self._actual_target_keys}")

        # --- Target Skalierung Validierung ---
        if self.target_scaling.method and self.target_scaling.method not in ['standardize']: # Aktuell nur standardize implementiert
             log_warning(f"Target-Skalierungsmethode '{self.target_scaling.method}' ist nicht implementiert. Ignoriere.")
             self.target_scaling.method = None
        if self.target_scaling.method:
             logging.info(f"Zielwert-Skalierung aktiviert: Methode='{self.target_scaling.method}'")

        # --- Workers ---
        if self.num_workers is None:
            self.num_workers = os.cpu_count() or 1 # Default auf alle CPUs
            logging.info(f"num_workers nicht gesetzt, verwende System-CPU-Anzahl: {self.num_workers}")
        elif self.num_workers < 1:
            log_warning(f"Ungültige num_workers ({self.num_workers}), setze auf 1 (sequentiell).")
            self.num_workers = 1
        if self.num_workers == 1:
             logging.info("Verarbeitung wird sequentiell durchgeführt (num_workers=1).")
        else:
             logging.info(f"Verarbeitung wird parallel durchgeführt mit {self.num_workers} Workern.")


        # --- Hash Berechnung ---
        # Hash basiert auf der *ursprünglichen* Konfiguration (aus YAML),
        # bevor interne Felder oder dynamische Parameter hinzugefügt werden.
        # Das laden der Konfig muss das dict übergeben.
        # Hier wird er nur initialisiert, der echte Hash kommt vom Lader.
        self._config_hash = "pending" # Platzhalter

        logging.debug("QM9Config Validierung und Initialisierung abgeschlossen.")

# ==============================================================================
# Block 5: Worker Funktion & Kern-Konvertierung
# ==============================================================================

# Diese Funktion läuft in einem separaten Prozess!
# Sie darf keine globalen Variablen (außer Konstanten) verwenden oder modifizieren.
# Logging muss über Queues oder spezielle Handler erfolgen (hier vereinfacht, gibt Fehler zurück).
def worker_process_molecule(mol_data: Tuple[int, str]) -> ProcessingResult:
                                # (index, mol_block_string)
    """
    Verarbeitet einen einzelnen Molekül-Block-String in einem Worker-Prozess.

    Args:
        mol_data: Tupel mit (Molekül-Index, SDF-Block-String).

    Returns:
        ProcessingResult: Enthält Erfolg/Misserfolg, Index und Ergebnis (Data oder Fehlerinfo).
    """
    global thread_local_config # Zugriff auf thread-lokale Kopie der Konfig

    index, mol_block = mol_data
    mol = None
    canonical_smiles = "[SMILES nicht verfügbar]"

    try:
        # Molekül aus SDF-Block lesen
        # Wichtig: RDKit braucht einen String-basierten Input hier
        mol = Chem.MolFromMolBlock(mol_block,
                                   removeHs=(not thread_local_config.use_h_atoms),
                                   sanitize=True)

        if mol is None:
            raise ValueError("RDKit konnte MolFromMolBlock nicht ausführen.")
        if mol.GetNumAtoms() == 0:
            raise ValueError("Molekül hat 0 Atome.")

        try:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass # Fehler beim SMILES generieren ist nicht kritisch für Verarbeitung

        # --- 1. Konformation ---
        conformer = mol.GetConformer()
        positions = conformer.GetPositions()
        pos_tensor = torch.tensor(positions, dtype=torch.float)
        if pos_tensor.shape[0] != mol.GetNumAtoms():
            raise ValueError(f"Positions-Anzahl ({pos_tensor.shape[0]}) != Atom-Anzahl ({mol.GetNumAtoms()}).")

        # --- 2. Atom-Features ---
        num_atoms = mol.GetNumAtoms()
        atom_features_list = []
        atomic_numbers = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in thread_local_config._atom_symbol_map:
                raise ValueError(f"Unerlaubtes Atom-Symbol: '{symbol}'.")
            atomic_numbers.append(atom.GetAtomicNum())

            onehot = torch.zeros(len(thread_local_config.allowed_atom_symbols), dtype=torch.float)
            onehot[thread_local_config._atom_symbol_map[symbol]] = 1.0
            features = [onehot]
            # Optionale Features... (Code hier vereinfacht, Logik wie im Original)
            if thread_local_config.add_atomic_mass: features.append(torch.tensor([atom.GetMass()], dtype=torch.float))
            # ... andere optionale Features ...
            atom_features_list.append(torch.cat(features))
        x_tensor = torch.stack(atom_features_list, dim=0)

        # --- 3. Kanten-Features & Indizes ---
        edge_indices_list, edge_features_list = [], []
        if num_atoms > 1:
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                if bond_type not in thread_local_config._bond_type_map:
                    raise ValueError(f"Unerlaubter Bindungstyp: '{BOND_TYPE_TO_NAME.get(bond_type, str(bond_type))}'.")

                onehot = torch.zeros(len(thread_local_config._allowed_bond_types_internal), dtype=torch.float)
                onehot[thread_local_config._bond_type_map[bond_type]] = 1.0
                features = [onehot]
                # ... andere optionale Bindungs-Features ...
                vec = torch.cat(features)
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices_list.extend([(start, end), (end, start)])
                edge_features_list.extend([vec, vec])

        if edge_indices_list:
            edge_index = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_features_list, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            # TODO: Korrekte Dimenstion für leere edge_attr berechnen
            expected_edge_dim = len(thread_local_config._allowed_bond_types_internal) # + optionale
            edge_attr = torch.empty((0, expected_edge_dim), dtype=torch.float)

        # --- 4. Zielwerte ---
        target_values = []
        props = mol.GetPropsAsDict()
        valid_targets = True
        for key in thread_local_config._actual_target_keys:
            if key not in props:
                raise ValueError(f"Fehlender Zielwert: '{key}'.")
            try:
                val = float(props[key])
                if math.isnan(val) or math.isinf(val):
                     # NaN/Inf in Targets können Probleme machen, Option: Überspringen oder als Fehler behandeln
                     raise ValueError(f"Ungültiger Zielwert (NaN/Inf) für '{key}'.")
                target_values.append(val)
            except (ValueError, TypeError):
                raise ValueError(f"Zielwert '{key}'='{props.get(key)}' nicht in Float konvertierbar.")

        y_tensor = torch.tensor(target_values, dtype=torch.float) # Shape [num_targets]

        # --- 5. Zielwert-Skalierung (im Worker anwenden) ---
        if thread_local_config.target_scaling.method == 'standardize':
            scaled_targets = []
            params = thread_local_config.target_scaling.params # Berechnete Params
            for i, key in enumerate(thread_local_config._actual_target_keys):
                if key in params:
                    mean = params[key]['mean']
                    std = params[key]['std']
                    # Vermeide Division durch Null (oder sehr kleine Zahl)
                    scaled_val = (y_tensor[i] - mean) / (std + 1e-8)
                    scaled_targets.append(scaled_val)
                else:
                    # Sollte nicht passieren, wenn Params korrekt berechnet wurden
                    scaled_targets.append(y_tensor[i]) # Unskaliert lassen
            y_tensor = torch.stack(scaled_targets) # Ersetze mit skalierten Werten

        y_tensor = y_tensor.unsqueeze(0) # Shape [1, num_targets]

        # --- 6. Data Objekt ---
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, pos=pos_tensor,
                    y=y_tensor, smiles=canonical_smiles,
                    atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
                    num_nodes=num_atoms)

        # --- 7. Validierung ---
        # data.validate() kann Warnungen loggen, was im Worker schwierig ist.
        # Führe grundlegende manuelle Checks durch oder ignoriere hier.
        # if not data.validate(raise_on_error=False):
        #     raise ValueError(f"PyG Data Validierung fehlgeschlagen: {getattr(data, 'validation_errors', 'N/A')}")

        return ProcessingResult(success=True, identifier=index, payload=data)

    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"Fehler bei Index {index} (SMILES: {canonical_smiles}): {str(e)}"
        # log_error(f"Worker Fehler: {error_msg}", exc_info=True) # Logging im Worker ist komplex
        return ProcessingResult(success=False, identifier=index, payload=(error_type, error_msg))

# Globale Variable für Konfiguration im Worker (wird bei Initialisierung gesetzt)
thread_local_config = None

def init_worker(config: QM9Config):
    """Initialisiert jeden Worker-Prozess (setzt die globale Konfig)."""
    global thread_local_config
    thread_local_config = config
    # Optional: RDKit Logger hier pro Worker deaktivieren (sollte aber global reichen)
    # Optional: Logging für Worker konfigurieren (z.B. mit QueueHandler)
    # print(f"Worker {os.getpid()} initialisiert.") # Debug

# ==============================================================================
# Block 6: Hauptklasse für den QM9 InMemoryDataset
# ==============================================================================
class QM9EnhancedDataset(InMemoryDataset):
    """
    Erweiterte Klasse zum Laden und Verarbeiten des QM9-Datensatzes als PyG InMemoryDataset.
    Nutzt parallele Verarbeitung und lädt Konfiguration aus YAML.
    """
    def __init__(self,
                 config: QM9Config,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        self.config = config # Die geladene und validierte Konfiguration
        self._raw_sdf_path = osp.join(self.raw_dir, self.config.raw_sdf_name)
        self._processed_base_dir = osp.join(self.root, 'processed')
        self._processed_version_dir = osp.join(self._processed_base_dir, self.config.version_name)
        self._processed_file_path = osp.join(self.processed_dir, f'data_{self.config._config_hash}.pt')
        self._metadata_file_path = osp.splitext(self._processed_file_path)[0] + '_meta.json'
        self._error_file_path = osp.splitext(self._processed_file_path)[0] + '_errors.json'

        logging.info(f"Initialisiere QM9EnhancedDataset in root: {self.root}")
        logging.info(f"Version: {self.config.version_name}, Hash: {self.config._config_hash}")

        # WICHTIG: Rufe super().__init__ auf.
        # root wird hier benötigt, processed_dir etc. werden über Properties gesteuert.
        super().__init__(root=self.root, # Basisverzeichnis
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)

        # Lade die Daten nach der Initialisierung (entweder aus Datei oder durch process())
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            logging.info(f"Datensatz erfolgreich geladen aus: {self.processed_paths[0]}")
            # Lade auch Metadaten, falls vorhanden und benötigt (z.B. für Skalierung)
            if osp.exists(self._metadata_file_path):
                 try:
                     with open(self._metadata_file_path, 'r') as f:
                         meta = json.load(f)
                         # Lade Skalierungsparameter zurück in die Konfig, falls sie beim Laden gebraucht werden
                         # (Normalerweise braucht man sie zum Ent-skalieren oder für neue Daten)
                         if 'target_scaling_params' in meta:
                              self.config.target_scaling.params = meta['target_scaling_params']
                              logging.info("Zielwert-Skalierungsparameter aus Metadaten geladen.")
                 except Exception as e:
                      log_warning(f"Konnte Metadaten nicht laden oder Skalierungsparameter fehlen: {e}")

        except FileNotFoundError:
            logging.error(f"FEHLER: Verarbeitete Datei nicht gefunden: {self.processed_paths[0]}")
            raise
        except Exception as e:
            log_error(f"FEHLER: Unerwarteter Fehler beim Laden der verarbeiteten Datei: {e}", exc_info=True)
            raise

        # Split anwenden (falls konfiguriert)
        if self.config.load_split:
            self._apply_split() # Diese Methode bleibt weitgehend gleich

        logging.info(f"Initialisierung von QM9EnhancedDataset abgeschlossen. Datensatzgröße: {len(self)}")


    @property
    def root(self) -> str:
        # Überschreibe root, um sicherzustellen, dass es auf config.root_dir zeigt
        return self.config.root_dir

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        """Gibt das spezifische Verzeichnis für diese Version/Hash zurück."""
        os.makedirs(self._processed_version_dir, exist_ok=True)
        return self._processed_version_dir

    @property
    def raw_file_names(self) -> List[str]:
        return [self.config.raw_sdf_name]

    @property
    def processed_file_names(self) -> List[str]:
        """Gibt den *relativen* Pfad zur verarbeiteten Datei zurück (relativ zu processed_dir)."""
        # Wichtig: PyG erwartet hier nur den Dateinamen oder relativen Pfad innerhalb processed_dir
        # Der Hash ist schon im Pfad enthalten, den wir konstruieren
        # Korrektur: PyG braucht hier den relativen Pfad vom *root* zum File,
        # oder nur den Dateinamen wenn es direkt in processed_dir liegt.
        # Da wir Unterordner nutzen (version_name), muss der Pfad relativ sein.
        # return [osp.join(self.config.version_name, f'data_{self.config._config_hash}.pt')] # Relativ zu processed_dir_base
        # Sicherste Variante: Nur der Dateiname, PyG sucht dann in self.processed_dir
        return [f'data_{self.config._config_hash}.pt']

    # --- Download Methode (weitgehend unverändert) ---
    def download(self):
        """Lädt die Rohdaten herunter und extrahiert sie."""
        # (Code von vorheriger Version kann hier größtenteils übernommen werden)
        # Stelle sicher, dass das raw-Verzeichnis existiert
        os.makedirs(self.raw_dir, exist_ok=True)
        logging.info(f"Prüfe auf Rohdaten in: {self.raw_dir}")
        target_sdf_path = self._raw_sdf_path
        archive_path = osp.join(self.raw_dir, self.config.raw_archive_name)

        if osp.exists(target_sdf_path) and not self.config.force_download: # force_download kommt aus CLI/config
            logging.info(f"Rohdatei '{self.config.raw_sdf_name}' existiert bereits. Download übersprungen.")
            return

        # ... (Restliche Download- und Extraktionslogik wie zuvor) ...
        logging.info(f"Starte Download von: {self.config.qm9_url} nach {archive_path}")
        try:
            response = requests.get(self.config.qm9_url, stream=True, timeout=120)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Download", disable=not TQDM_AVAILABLE, leave=False)
            with open(archive_path, 'wb') as f, progress_bar:
                 for chunk in response.iter_content(chunk_size=8192):
                     f.write(chunk); progress_bar.update(len(chunk))
            if total_size != 0 and progress_bar.n != total_size: raise IOError("Download unvollständig.")
            logging.info("Download erfolgreich.")

            logging.info(f"Extrahiere '{self.config.raw_sdf_name}'...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                 sdf_member = next((m for m in tar.getmembers() if m.name.endswith(self.config.raw_sdf_name)), None)
                 if sdf_member is None: raise FileNotFoundError("SDF nicht im Archiv.")
                 sdf_member.name = osp.basename(sdf_member.name)
                 tar.extract(sdf_member, path=self.raw_dir)
            if not osp.exists(target_sdf_path): raise FileNotFoundError("Extraktion fehlgeschlagen.")
            logging.info(f"Extraktion erfolgreich: '{target_sdf_path}'.")

            logging.info(f"Entferne Archivdatei: {archive_path}")
            os.remove(archive_path)

        except Exception as e:
            log_error(f"Download/Extraktion fehlgeschlagen: {e}", exc_info=True)
            # Cleanup
            if osp.exists(archive_path): try: os.remove(archive_path) catch: pass
            if osp.exists(target_sdf_path): try: os.remove(target_sdf_path) catch: pass
            raise RuntimeError("Daten-Download/Extraktion fehlgeschlagen.") from e


    # --- Haupt-Verarbeitungsmethode (stark überarbeitet) ---
    def process(self):
        """
        Verarbeitet die rohe SDF-Datei parallel und erstellt die PyG Data-Objekte.
        """
        logging.info("=" * 60)
        logging.info(f" Starte Verarbeitung (parallel={self.config.num_workers > 1}) ".center(60, "="))
        logging.info("=" * 60)
        start_time = time.time()

        # --- 1. Lese SDF-Molekülblöcke ---
        logging.info(f"Lese Molekülblöcke aus: {self._raw_sdf_path}")
        mol_blocks = []
        try:
            # Lese die Datei und teile sie in einzelne Molekülblöcke auf
            with open(self._raw_sdf_path, 'r') as f:
                content = f.read()
            # SDF-Blöcke sind durch "$$$$" getrennt
            raw_blocks = content.split('$$$$\n')
            mol_blocks = [(i, block.strip()) for i, block in enumerate(raw_blocks) if block.strip()] # Index + Block
            num_total_entries = len(mol_blocks)
            logging.info(f"{num_total_entries} Moleküleinträge gefunden.")
            if num_total_entries == 0:
                raise ValueError("SDF enthält keine Moleküleinträge.")
        except Exception as e:
            log_error(f"Fehler beim Lesen oder Aufteilen der SDF-Datei: {e}", exc_info=True)
            raise

        # --- 2. Optional: Berechne Target-Skalierungsparameter ---
        if self.config.target_scaling.method == 'standardize':
            logging.info("Berechne Standardisierungs-Parameter für Zielwerte...")
            all_targets_raw = defaultdict(list)
            # Schneller sequentieller Durchlauf nur zum Extrahieren der Targets
            temp_config_no_scale = self.config # TODO: Sicherstellen, dass keine Skalierung angewendet wird
            for i, mol_block in tqdm(mol_blocks, desc="Target-Extraktion", disable=not TQDM_AVAILABLE):
                 try:
                     mol = Chem.MolFromMolBlock(mol_block, removeHs=(not self.config.use_h_atoms), sanitize=True)
                     if mol:
                         props = mol.GetPropsAsDict()
                         valid_mol_targets = True
                         for key in self.config._actual_target_keys:
                             try:
                                 val = float(props[key])
                                 if math.isnan(val) or math.isinf(val):
                                     valid_mol_targets = False; break
                                 all_targets_raw[key].append(val)
                             except (KeyError, ValueError, TypeError):
                                 valid_mol_targets = False; break
                 except Exception:
                     pass # Ignoriere Fehler hier, nur gültige Targets sammeln

            # Berechne Mittelwert und Standardabweichung
            scaling_params = {}
            for key, values in all_targets_raw.items():
                 if values:
                     tensor_vals = torch.tensor(values, dtype=torch.float)
                     mean = torch.mean(tensor_vals).item()
                     std = torch.std(tensor_vals).item()
                     if std < 1e-8: # Vermeide Division durch Null bei konstanten Werten
                          log_warning(f"Standardabweichung für Target '{key}' ist nahe Null. Skalierung wird diesen Wert evtl. nicht ändern.")
                          std = 1.0 # Setze auf 1, um Division durch Null zu vermeiden
                     scaling_params[key] = {'mean': mean, 'std': std}
                 else:
                     log_warning(f"Keine gültigen Werte für Target '{key}' gefunden. Kann nicht skaliert werden.")

            self.config.target_scaling.params = scaling_params # Speichere berechnete Params
            logging.info(f"Standardisierungs-Parameter berechnet: {scaling_params}")


        # --- 3. Parallele Verarbeitung ---
        logging.info(f"Starte Molekülverarbeitung mit {self.config.num_workers} Worker(n)...")
        processed_results: List[ProcessingResult] = []

        # ProcessPoolExecutor ist oft einfacher zu handhaben als Pool
        # initializer und initargs zum Übergeben der Konfiguration an jeden Worker
        with ProcessPoolExecutor(max_workers=self.config.num_workers,
                                 initializer=init_worker,
                                 initargs=(self.config,)) as executor:
            # Verwende executor.map für einfachere Abarbeitung mit Fortschrittsbalken
            # map behält die Reihenfolge bei, was gut ist, wenn der Index wichtig ist
            # chunksize steuert, wie viele Aufgaben pro Worker auf einmal geholt werden
            futures = executor.map(worker_process_molecule, mol_blocks, chunksize=self.config.processing_chunksize)

            # Fortschritt anzeigen und Ergebnisse sammeln
            results_iterator = tqdm(futures, total=num_total_entries, desc="Verarbeitung", disable=not TQDM_AVAILABLE)
            for result in results_iterator:
                processed_results.append(result)

        # --- 4. Ergebnisse sammeln und filtern ---
        logging.info("Sammle und filtere Ergebnisse...")
        valid_data_list: List[Data] = []
        error_list: List[Dict] = []
        seen_smiles: Set[str] = set()
        processed_ok_count = 0
        duplicate_count = 0
        error_count = 0

        for result in processed_results:
            if result.success:
                data = result.payload
                # Duplikat-Check (nach der Parallelisierung)
                if self.config.check_duplicates:
                    smiles = data.smiles
                    if smiles in seen_smiles:
                        duplicate_count += 1
                        continue # Überspringe Duplikat
                    else:
                        seen_smiles.add(smiles)

                # Pre-Filter / Pre-Transform (hier anwenden, da sie auf Data operieren)
                passes_filter = True
                if self.pre_filter is not None and not self.pre_filter(data):
                     passes_filter = False
                if passes_filter and self.pre_transform is not None:
                     try:
                         data = self.pre_transform(data)
                         if data is None: passes_filter = False
                     except Exception as e:
                         log_warning(f"Fehler bei pre_transform für {getattr(data, 'smiles', result.identifier)}: {e}")
                         passes_filter = False

                if passes_filter:
                     valid_data_list.append(data)
                     processed_ok_count += 1
                else:
                     # Als Fehler behandeln, wenn Filter/Transform fehlschlägt oder None zurückgibt
                     error_count += 1
                     error_list.append({
                         'identifier': result.identifier,
                         'smiles': getattr(data, 'smiles', '[N/A]'),
                         'error_type': 'FilterTransformError',
                         'error_message': 'Von pre_filter oder pre_transform verworfen/fehlerhaft.'
                     })

            else:
                # Fehler im Worker aufgetreten
                error_count += 1
                error_type, error_msg = result.payload
                error_list.append({
                    'identifier': result.identifier,
                    'smiles': '[Fehler vor SMILES-Generierung]', # SMILES oft nicht verfügbar bei Fehlern
                    'error_type': error_type,
                    'error_message': error_msg
                })

        # --- 5. Verarbeitungszusammenfassung ---
        duration = time.time() - start_time
        logging.info("-" * 60)
        logging.info(" Verarbeitung abgeschlossen ".center(60, "-"))
        logging.info(f"Dauer: {duration:.2f} Sekunden")
        logging.info(f"Ursprüngliche Einträge: {num_total_entries}")
        logging.info(f"Erfolgreich verarbeitet (nach Filter/Transform): {processed_ok_count}")
        if self.config.check_duplicates:
            logging.info(f"Übersprungene Duplikate: {duplicate_count}")
        logging.info(f"Fehlerhafte/Verworfene Einträge: {error_count}")
        logging.info("-" * 60)

        # --- 6. Wichtige Prüfung: Ist die Liste leer? ---
        if not valid_data_list:
            log_error("FEHLER: Nach der Verarbeitung wurden keine gültigen Datenobjekte erstellt.")
            self._save_errors(error_list) # Speichere zumindest die Fehler
            raise ValueError("Verarbeitung ergab einen leeren Datensatz. Abbruch.")

        # --- 7. Daten kollationieren und speichern ---
        logging.info(f"Kollationiere {len(valid_data_list)} gültige Datenobjekte...")
        try:
            data, slices = self.collate(valid_data_list)
        except Exception as e:
            log_error(f"Fehler beim Kollationieren der Daten: {e}", exc_info=True)
            raise RuntimeError("Datenkollationierung fehlgeschlagen.") from e

        logging.info(f"Speichere verarbeitete Daten nach: {self.processed_paths[0]}")
        try:
            torch.save((data, slices), self.processed_paths[0]) # Nutzen von processed_paths Property
            logging.info("Speichern der verarbeiteten Daten abgeschlossen.")
        except Exception as e:
            log_error(f"FEHLER beim Speichern der verarbeiteten Daten: {e}", exc_info=True)
            if osp.exists(self.processed_paths[0]): try: os.remove(self.processed_paths[0]) catch: pass
            raise RuntimeError("Speichern der verarbeiteten Daten fehlgeschlagen.") from e

        # --- 8. Metadaten und Fehler speichern ---
        self._save_metadata(processed_ok_count, duplicate_count, error_count, duration)
        self._save_errors(error_list)

        logging.info("=" * 60)
        logging.info(" Verarbeitungsprozess erfolgreich beendet ".center(60, "="))
        logging.info("=" * 60)


    def _save_metadata(self, num_processed: int, num_duplicates: int, num_errors: int, duration: float):
        """Speichert Metadaten über den Verarbeitungslauf als JSON-Datei."""
        logging.info(f"Speichere Metadaten nach: {self._metadata_file_path}...")

        # Sammle Versionen etc. (wie zuvor)
        try: rdkit_version = Chem.__version__ catch: rdkit_version = "N/A"
        try: pyg_version = torch_geometric.__version__ catch: pyg_version = "N/A"
        try: import platform; py_version = platform.python_version() catch: py_version = "N/A"

        # Konfiguration für Metadaten vorbereiten (ohne interne Felder)
        serializable_config = {k: v for k, v in asdict(self.config).items() if not k.startswith('_')}
        # TargetScalingConfig muss eventuell speziell behandelt werden, wenn nicht serialisierbar
        if 'target_scaling' in serializable_config and isinstance(serializable_config['target_scaling'], TargetScalingConfig):
             # Nur Methode speichern, Params kommen separat
             serializable_config['target_scaling'] = {'method': self.config.target_scaling.method}


        # Feature-Dimensionen (aus dem ersten gültigen Datenobjekt)
        feature_dims = {'node': -1, 'edge': -1}
        if self.data and hasattr(self.data, 'x') and hasattr(self.data, 'edge_attr'):
            try: feature_dims['node'] = self.data.x.shape[1] catch: pass
            try: feature_dims['edge'] = self.data.edge_attr.shape[1] catch: pass


        metadata = {
            "processing_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "processing_duration_seconds": round(duration, 2),
            "config_hash": self.config._config_hash,
            "qm9_config": serializable_config,
             # Füge die Skalierungsparameter separat hinzu
            "target_scaling_params": self.config.target_scaling.params if self.config.target_scaling.method else None,
            "dataset_statistics": {
                "successfully_processed": num_processed,
                "duplicates_skipped": num_duplicates,
                "errors_skipped": num_errors,
                "total_in_sdf": num_processed + num_duplicates + num_errors,
            },
            "feature_dimensions": feature_dims,
            "environment_versions": {
                "python": py_version, "pytorch": torch.__version__,
                "torch_geometric": pyg_version, "rdkit": rdkit_version,
            },
            "output_files": {
                "processed_data_file": osp.basename(self.processed_paths[0]),
                "metadata_file": osp.basename(self._metadata_file_path),
                "error_file": osp.basename(self._error_file_path),
            },
            "source_data": {
                "sdf_filename": self.config.raw_sdf_name, "download_url": self.config.qm9_url,
            }
        }

        try:
            with open(self._metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, default=str) # default=str als Fallback
            logging.info("Metadaten erfolgreich gespeichert.")
        except Exception as e:
            log_error(f"Fehler beim Speichern der Metadaten: {e}", exc_info=True)


    def _save_errors(self, error_list: List[Dict]):
        """Speichert die Liste der aufgetretenen Fehler als JSON-Datei."""
        if not error_list:
            logging.info("Keine Verarbeitungsfehler aufgetreten.")
            # Lösche alte Fehlerdatei, falls vorhanden
            if osp.exists(self._error_file_path):
                 try: os.remove(self._error_file_path) catch: pass
            return

        logging.warning(f"Speichere {len(error_list)} Verarbeitungsfehler nach: {self._error_file_path}...")
        try:
            # Fasse Fehler nach Typ zusammen für eine bessere Übersicht
            error_summary = defaultdict(int)
            for err in error_list:
                error_summary[err.get('error_type', 'Unknown')] += 1

            error_data = {
                "total_errors": len(error_list),
                "error_summary_by_type": dict(error_summary),
                "error_details": error_list # Liste aller Fehler
            }
            with open(self._error_file_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=4, default=str)
            logging.info("Fehler erfolgreich gespeichert.")
        except Exception as e:
            log_error(f"Fehler beim Speichern der Fehlerdatei: {e}", exc_info=True)

    def _apply_split(self):
        """Filtert geladene Daten basierend auf Split-Definition (unverändert)."""
        # (Code von vorheriger Version kann hier übernommen werden)
        if not self.config.load_split or not self.config.split_definition_path: return
        split_file_path = Path(self.config.split_definition_path)
        split_name = self.config.load_split
        split_key = f"{split_name}_idx"
        logging.info(f"Wende Split '{split_name}' an aus {split_file_path}...")
        if not split_file_path.is_file(): raise FileNotFoundError(f"Split-Datei nicht gefunden: {split_file_path}")
        try:
            split_dict = torch.load(split_file_path)
            if not isinstance(split_dict, dict): raise TypeError("Split-Datei enthält kein Dict.")
            indices = split_dict.get(split_key)
            if indices is None: raise ValueError(f"Schlüssel '{split_key}' nicht in Split-Datei gefunden.")
            if not isinstance(indices, torch.Tensor): indices = torch.tensor(indices, dtype=torch.long)
            if indices.dtype != torch.long: indices = indices.to(torch.long)

            num_total, num_split = len(self), len(indices)
            logging.info(f"Indizes für Split '{split_name}': {num_split} (aus {num_total} Samples)")
            if num_split == 0:
                 self.data, self.slices = self.collate([]); logging.warning("Split ist leer.")
            else:
                 max_idx, min_idx = indices.max().item(), indices.min().item()
                 if max_idx >= num_total or min_idx < 0: raise IndexError("Split-Indizes außerhalb des gültigen Bereichs.")
                 self.data, self.slices = self.index_select(indices)
                 logging.info(f"Datensatzgröße nach Split: {len(self)}")
                 if len(self) != num_split: logging.warning("Endgültige Größe != Anzahl Split-Indizes.")
        except Exception as e:
            log_error(f"Fehler beim Anwenden des Splits: {e}", exc_info=True)
            raise RuntimeError("Fehler beim Anwenden des Datensatz-Splits.") from e


# ==============================================================================
# Block 7: Hilfsfunktionen & Kommandozeilen-Interface
# ==============================================================================

def load_config_from_yaml(config_path: str) -> Tuple[QM9Config, str]:
    """Lädt die Konfiguration aus einer YAML-Datei und validiert sie."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML ist erforderlich, um Konfigurationen aus YAML zu laden. Bitte installieren (`pip install pyyaml`).")
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")

    logging.info(f"Lade Konfiguration aus: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                 raise TypeError("YAML-Datei enthält keine gültige Dictionary-Konfiguration.")

        # Erstelle QM9Config Instanz aus dem Dictionary
        # Trenne TargetScalingConfig separat
        target_scaling_dict = config_dict.pop('target_scaling', {})
        target_scaling_config = TargetScalingConfig(**target_scaling_dict)

        config = QM9Config(**config_dict, target_scaling=target_scaling_config)

        # Generiere Hash basierend auf dem *geladenen* Dictionary (vor __post_init__)
        # Schließe Pfade etc. aus, die nicht zum Inhalt beitragen
        hash_dict = {k: v for k, v in config_dict.items() if k not in ['root_dir', 'version_name', 'qm9_url', 'raw_archive_name', 'raw_sdf_name', 'num_workers', 'split_definition_path', 'load_split']}
        # Füge relevante Teile der Skalierung hinzu
        hash_dict['target_scaling_method'] = target_scaling_config.method
        config._config_hash = config.generate_config_hash(hash_dict)
        logging.info(f"Generierter Konfigurations-Hash: {config._config_hash}")

        # __post_init__ wird automatisch aufgerufen und validiert weiter

        return config

    except yaml.YAMLError as e:
        log_error(f"Fehler beim Parsen der YAML-Konfigurationsdatei: {e}", exc_info=True)
        raise ValueError(f"Ungültige YAML-Datei: {config_path}") from e
    except (TypeError, ValueError) as e:
        log_error(f"Fehler in der Konfigurationsstruktur oder den Werten: {e}", exc_info=True)
        raise ValueError(f"Ungültige Konfiguration in {config_path}") from e


def main():
    """Hauptfunktion: Parsen von Argumenten, Laden der Konfig, Starten der Verarbeitung."""
    parser = argparse.ArgumentParser(description="QM9 Datensatzverarbeitung mit PyG.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Pfad zur YAML-Konfigurationsdatei."
    )
    parser.add_argument(
        "--force_process",
        action="store_true",
        help="Verarbeitung erzwingen, auch wenn verarbeitete Datei existiert."
    )
    parser.add_argument(
        "--force_download",
         action="store_true",
         help="Download der Rohdaten erzwingen, auch wenn sie existieren."
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging-Level setzen."
    )
    parser.add_argument(
        "--load_split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Optional: Welchen Split nach der Verarbeitung laden (benötigt split_definition_path in YAML)."
    )

    args = parser.parse_args()

    # --- Logging Setup ---
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(process)d] - %(message)s', # Prozess-ID hinzugefügt
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if RDKIT_AVAILABLE: logging.info("RDKit Logger deaktiviert.")
    logging.info(f"Logging-Level gesetzt auf: {args.loglevel}")

    # --- Konfiguration laden ---
    try:
        config = load_config_from_yaml(args.config_path)
        # Überschreibe Konfig-Werte mit CLI-Argumenten, falls vorhanden
        config.force_download = args.force_download # Direkt in config speichern (nicht Teil des Hash)
        config.force_process = args.force_process   # Direkt in config speichern (nicht Teil des Hash)
        config.load_split = args.load_split         # Direkt in config speichern (nicht Teil des Hash)

    except (ImportError, FileNotFoundError, ValueError) as e:
        log_error(f"Fehler beim Laden der Konfiguration: {e}")
        return # Beenden bei Konfigurationsfehler

    # --- Dataset Instanziierung (löst Download/Prozess aus) ---
    logging.info("Instanziiere QM9EnhancedDataset...")
    try:
        start_time = time.time()
        # Wichtig: Die force_process Logik ist in PyG's InMemoryDataset eingebaut.
        # Wenn die Datei existiert und force_process=False ist, wird process() nicht aufgerufen.
        qm9_dataset = QM9EnhancedDataset(config=config)
        end_time = time.time()
        logging.info(f"Dataset-Initialisierung dauerte {end_time - start_time:.2f} Sekunden.")

        # --- Inspektion ---
        logging.info("=" * 60)
        logging.info(" Datensatz Inspektion ".center(60, "="))
        logging.info(f"Anzahl Graphen: {len(qm9_dataset)}")
        if len(qm9_dataset) > 0:
            first_data = qm9_dataset[0]
            logging.info(f"Erstes Element (Index 0):")
            logging.info(f"  SMILES: {first_data.smiles}")
            logging.info(f"  Knoten: {first_data.num_nodes}, Kanten: {first_data.num_edges}")
            logging.info(f"  x.shape: {first_data.x.shape}")
            logging.info(f"  edge_attr.shape: {first_data.edge_attr.shape if first_data.edge_attr is not None else 'N/A'}")
            logging.info(f"  pos.shape: {first_data.pos.shape}")
            logging.info(f"  y.shape: {first_data.y.shape}")
            logging.info(f"  y Werte (erste 5): {first_data.y[0, :5].tolist()}")
            # Überprüfe, ob y-Werte skaliert wurden (sollten um 0 liegen bei standardize)
            if config.target_scaling.method == 'standardize':
                 logging.info(f"  y Mittelwert (ca.): {first_data.y.mean().item():.4f}")

            # Überprüfe Metadaten und Fehlerdatei Existenz
            if osp.exists(qm9_dataset._metadata_file_path):
                 logging.info(f"Metadaten-Datei gefunden: {qm9_dataset._metadata_file_path}")
            if osp.exists(qm9_dataset._error_file_path):
                 logging.warning(f"Fehler-Datei gefunden: {qm9_dataset._error_file_path}")

        else:
            logging.warning("Datensatz ist leer nach Verarbeitung/Laden.")
        logging.info("=" * 60)

    except Exception as e:
        log_error(f"Ein unerwarteter Fehler ist im Hauptablauf aufgetreten.", exc_info=True)


# ==============================================================================
# Block 8: Skriptausführung
# ==============================================================================
if __name__ == '__main__':
    # Setze Startmethode für Multiprocessing (wichtig für manche Systeme)
    # 'fork' kann Probleme mit RDKit/PyTorch machen, 'spawn' ist sicherer
    try:
        mp.set_start_method('spawn', force=True) # 'force=True' falls schon gesetzt
        logging.debug("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logging.debug("Multiprocessing start method already set.")
    except ValueError:
         log_warning("Konnte Multiprocessing start method nicht auf 'spawn' setzen.")


    main()
    logging.info("Skriptausführung beendet.")
