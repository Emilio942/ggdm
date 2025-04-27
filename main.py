# -*- coding: utf-8 -*-
"""
Modul zur Verarbeitung des QM9-Datensatzes in PyTorch Geometric Data-Objekte.

Dieses Skript lädt den QM9-Datensatz herunter, verarbeitet die SDF-Datei,
extrahiert Molekülstrukturen und Zielwerte und speichert sie in einem
PyTorch Geometric InMemoryDataset-Format. Es legt Wert auf klare
Konfiguration, kontrollierte Protokollierung und sequentielle Verarbeitung
zur einfacheren Fehlersuche und Stabilität.
"""

# ==============================================================================
# Block 1: Import von Abhängigkeiten
# ==============================================================================
import os
import os.path as osp
import tarfile         # Zum Entpacken von .tar.gz-Archiven
import requests        # Zum Herunterladen von Dateien über HTTP
import shutil          # Für Dateioperationen wie das Verschieben
import logging         # Für die Protokollierung von Ereignissen und Fehlern
import traceback       # Zum Formatieren von Fehler-Stacktraces
import hashlib         # Zur Erzeugung von Hashes (z.B. für Konfigurationen)
import json            # Zum Lesen/Schreiben von Metadaten (JSON-Format)
import time            # Zur Zeitmessung von Prozessen
# import sys           # Nicht mehr benötigt
from typing import List, Dict, Tuple, Optional, Callable, Any, Set, Union
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from collections import defaultdict

# PyTorch und PyG Imports
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.typing import Tensor

# RDKit Imports
try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT
    # from rdkit.Chem import Descriptors # Beispiel für optionale, hier nicht verwendete Features
    from rdkit import RDLogger
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    # Platzhalter, falls RDKit nicht installiert ist (führt später zu Fehlern, aber ermöglicht Import)
    BT = type('BondType', (), {'SINGLE': None, 'DOUBLE': None, 'TRIPLE': None, 'AROMATIC': None})
    logging.error("RDKit konnte nicht importiert werden. Die Verarbeitung wird fehlschlagen.")
    # Beende das Skript nicht sofort, damit die Konfiguration etc. noch definiert werden kann

# TQDM für Fortschrittsbalken
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Einfache Platzhalterfunktion, falls tqdm nicht verfügbar ist
    def tqdm(iterable, *args, **kwargs):
        logging.info("tqdm nicht verfügbar. Fortschrittsbalken werden nicht angezeigt.")
        return iterable

# Multiprocessing (obwohl hier sequentiell verwendet, bleiben Imports für potenzielle Erweiterungen)
from multiprocessing import Pool, cpu_count, current_process


# ==============================================================================
# Block 2: Konfiguration der Protokollierung (Logging)
# ==============================================================================
# Ziel: Informatives Logging, aber Vermeidung von Überflutung des Terminals

# Maximale Anzahl von Meldungen pro Warnungstyp, um Wiederholungen zu begrenzen
MAX_LOG_PER_WARNING_TYPE: int = 5
# Zähler für die verschiedenen Warnungstypen
warning_counters: Dict[str, int] = defaultdict(int)

# Grundlegende Logging-Konfiguration
# Startet mit INFO-Level, um den allgemeinen Fortschritt anzuzeigen
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# RDKit-Logger deaktivieren, um übermäßige Ausgaben zu vermeiden
if RDKIT_AVAILABLE:
    RDLogger.DisableLog('rdApp.*')
    logging.info("RDKit Logger erfolgreich deaktiviert.")
else:
    logging.warning("RDKit nicht verfügbar, Logger konnte nicht deaktiviert werden.")

# Hilfsfunktion für limitierte Warnmeldungen
def log_limited_warning(message: str, key: Optional[str] = None):
    """
    Protokolliert eine Warnmeldung, begrenzt aber die Anzahl der Meldungen pro Typ.

    Args:
        message (str): Die zu protokollierende Nachricht.
        key (Optional[str]): Ein optionaler Schlüssel zur Gruppierung ähnlicher Warnungen.
                              Wenn None, wird die Nachricht selbst als Schlüssel verwendet.
    """
    global warning_counters, MAX_LOG_PER_WARNING_TYPE
    log_key = key if key else message  # Eindeutiger Schlüssel für diese Warnungsart
    warning_counters[log_key] += 1

    if warning_counters[log_key] <= MAX_LOG_PER_WARNING_TYPE:
        logging.warning(message)
    elif warning_counters[log_key] == MAX_LOG_PER_WARNING_TYPE + 1:
        # Informieren, dass weitere Meldungen dieses Typs unterdrückt werden
        logging.warning(f"Weitere Warnungen des Typs '{log_key}' werden unterdrückt...")


# ==============================================================================
# Block 3: Konstanten und Mappings
# ==============================================================================
# Mappings zwischen RDKit BondType Enum und String-Namen für die Konfiguration
BOND_TYPE_TO_NAME: Dict[BT, str] = {
    BT.SINGLE: 'SINGLE',
    BT.DOUBLE: 'DOUBLE',
    BT.TRIPLE: 'TRIPLE',
    BT.AROMATIC: 'AROMATIC'
}
# Umgekehrtes Mapping für die interne Verwendung
NAME_TO_BOND_TYPE: Dict[str, BT] = {v: k for k, v in BOND_TYPE_TO_NAME.items()}


# ==============================================================================
# Block 4: Konfigurations-Datenklasse (QM9Config)
# ==============================================================================
@dataclass
class QM9Config:
    """
    Konfigurations-Datenklasse für die Verarbeitung des QM9-Datensatzes.

    Diese Klasse bündelt alle Einstellungen, die den Download, die Verarbeitung
    und die Feature-Extraktion steuern.
    """
    # --- Pfade und Versionierung ---
    root: str = osp.join(os.getcwd(), 'data', 'QM9_FinalFix_Sequential') # Eindeutiger Root für diese Version
    version_name: str = "v7_seq_detailed" # Eindeutige Versionierung für verarbeitete Daten

    # --- Molekülverarbeitung ---
    use_h_atoms: bool = True # Wasserstoffatome im Molekül belassen?
    allowed_atom_symbols: List[str] = field(default_factory=lambda: ['H', 'C', 'N', 'O', 'F']) # Erlaubte Atomsymbole
    allowed_bond_type_names: List[str] = field(default_factory=lambda: ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']) # Erlaubte Bindungstypen (Namen)

    # --- Optionale Features (Standardmäßig deaktiviert für Einfachheit) ---
    add_atomic_mass: bool = False         # Atommasse als Knoten-Feature hinzufügen?
    add_formal_charge: bool = False       # Formalladung als Knoten-Feature hinzufügen?
    add_hybridization: bool = False       # Hybridisierung als Knoten-Feature hinzufügen? (One-Hot)
    add_is_aromatic_atom: bool = False    # Ist Atom aromatisch? (Knoten-Feature)
    add_is_in_ring_atom: bool = False     # Ist Atom in einem Ring? (Knoten-Feature)
    add_is_conjugated_bond: bool = False  # Ist Bindung konjugiert? (Kanten-Feature)
    add_is_in_ring_bond: bool = False     # Ist Bindung in einem Ring? (Kanten-Feature)

    # --- Zielvariablen (Targets) ---
    # **********************************************************************
    # *** KORRIGIERTE LISTE ALLER VERFÜGBAREN QM9 TARGET PROPERTIES      ***
    # **********************************************************************
    # Diese Liste enthält die 12 von RDKit direkt aus der SDF-Datei lesbaren
    # QM9-Eigenschaften. Die ursprünglichen 19 Eigenschaften erforderten
    # zusätzliche Berechnungen oder waren in der SDF anders benannt.
    all_target_properties: List[str] = field(default_factory=lambda: [
        'mu',    # Dipolmoment
        'alpha', # Isotrope Polarisierbarkeit
        'HOMO',  # Energie des höchsten besetzten Molekülorbitals
        'LUMO',  # Energie des niedrigsten unbesetzten Molekülorbitals
        'gap',   # Energielücke (LUMO - HOMO)
        'R2',    # Elektronische räumliche Ausdehnung
        'ZPVE',  # Nullpunkt-Schwingungsenergie
        'U0',    # Innere Energie bei 0 K
        'U',     # Innere Energie bei 298.15 K
        'H',     # Enthalpie bei 298.15 K
        'G',     # Freie Enthalpie bei 298.15 K
        'CV'     # Wärmekapazität bei 298.15 K
    ])
    # **********************************************************************
    # **********************************************************************

    # Welche Targets sollen tatsächlich geladen und im `y`-Tensor gespeichert werden?
    # Wenn None, werden alle `all_target_properties` geladen.
    target_keys_to_load: Optional[List[str]] = None

    # --- Download- und Verarbeitungseinstellungen ---
    qm9_url: str = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz' # Download-URL
    raw_archive_name: str = 'gdb9.tar.gz' # Name der heruntergeladenen Archivdatei
    raw_sdf_name: str = 'gdb9.sdf'        # Name der SDF-Datei im Archiv
    force_download: bool = False          # Erneuten Download erzwingen, auch wenn Datei existiert?
    force_process: bool = False           # Erneute Verarbeitung erzwingen, auch wenn verarbeitete Datei existiert?
    num_workers: int = 1                  # Anzahl der Worker-Prozesse (SEQUENTIELL: auf 1 gesetzt!)
    check_duplicates: bool = True         # Duplikate anhand von kanonischem SMILES erkennen und überspringen?

    # --- Aufteilung (Splitting) ---
    # Pfad zur Datei, die die Indizes für Trainings-, Validierungs- und Testsets enthält (z.B. .pt-Datei von PyTorch)
    split_definition_path: Optional[str] = None
    # Welcher Split soll geladen werden? Muss ein Schlüssel in der split_definition_path-Datei sein (z.B. 'train', 'val', 'test')
    load_split: Optional[str] = None # z.B. 'train', 'val', 'test'

    # --- Interne, abgeleitete Attribute (nicht direkt setzen!) ---
    # Diese werden in __post_init__ berechnet.
    _atom_symbol_map: Dict[str, int] = field(init=False, repr=False) # Mapping Atomsymbol -> Index für One-Hot
    _allowed_bond_types_internal: List[BT] = field(init=False, repr=False) # Erlaubte RDKit BondType Enums
    _bond_type_map: Dict[BT, int] = field(init=False, repr=False) # Mapping RDKit BondType -> Index für One-Hot
    _actual_target_keys: List[str] = field(init=False, repr=False) # Tatsächlich zu ladende Target-Schlüssel
    _config_hash: str = field(init=False, repr=False) # Hash der relevanten Konfiguration für Dateinamen

    def __post_init__(self):
        """
        Validiert die Konfiguration und berechnet abgeleitete Attribute nach der Initialisierung.
        """
        logging.debug("Starte QM9Config __post_init__...")

        # 1. Validiere und konvertiere erlaubte Bindungstypen
        self._allowed_bond_types_internal = []
        invalid_bond_names = [name for name in self.allowed_bond_type_names if name not in NAME_TO_BOND_TYPE]
        if invalid_bond_names:
            raise ValueError(f"Ungültige Bindungstyp-Namen in 'allowed_bond_type_names': {invalid_bond_names}. "
                             f"Gültige Typen sind: {list(NAME_TO_BOND_TYPE.keys())}")
        self._allowed_bond_types_internal = [NAME_TO_BOND_TYPE[name] for name in self.allowed_bond_type_names]
        logging.debug(f"Interne erlaubte RDKit-Bindungstypen: {self._allowed_bond_types_internal}")

        # 2. Erstelle Mapping-Dictionaries für One-Hot-Kodierung
        self._atom_symbol_map = {symbol: i for i, symbol in enumerate(self.allowed_atom_symbols)}
        self._bond_type_map = {bond_type: i for i, bond_type in enumerate(self._allowed_bond_types_internal)}
        logging.debug(f"Atom-Symbol-Map: {self._atom_symbol_map}")
        logging.debug(f"Bindungstyp-Map: {self._bond_type_map}")

        # 3. Bestimme die tatsächlich zu ladenden Zielwerte (Targets)
        available_properties_set = set(self.all_target_properties) # Nutzt die korrigierte Liste
        if self.target_keys_to_load is None:
            # Lade alle definierten 'all_target_properties'
            self._actual_target_keys = list(self.all_target_properties)
            logging.info("Es werden alle verfügbaren Zielwerte (Targets) geladen.")
        else:
            # Überprüfe, ob die angeforderten Targets gültig sind
            invalid_keys = [key for key in self.target_keys_to_load if key not in available_properties_set]
            if invalid_keys:
                raise ValueError(f"Ungültige Zielwert-Schlüssel in 'target_keys_to_load': {invalid_keys}. "
                                 f"Verfügbare Schlüssel sind: {self.all_target_properties}")
            self._actual_target_keys = list(self.target_keys_to_load)
            logging.info(f"Es werden folgende Zielwerte (Targets) geladen: {self._actual_target_keys}")

        # 4. Berechne einen Hash der relevanten Konfigurationsteile
        # Dieser Hash wird Teil des Dateinamens der verarbeiteten Daten,
        # um verschiedene Konfigurationen unterscheiden zu können.
        hash_relevant_fields = [
            f.name for f in fields(self) if f.init and f.name not in {
                # Felder, die den Inhalt der verarbeiteten Datei nicht beeinflussen
                'root', 'version_name', 'qm9_url', 'raw_archive_name', 'raw_sdf_name',
                'force_download', 'force_process', 'num_workers',
                'split_definition_path', 'load_split'
            }
        ]
        # Stelle sicher, dass Listen und andere nicht-primitive Typen konsistent gehasht werden
        hash_config_dict = {k: getattr(self, k) for k in hash_relevant_fields}
        # Konvertiere das Dict in einen sortierten JSON-String für einen stabilen Hash
        config_string = json.dumps(hash_config_dict, sort_keys=True, default=str)
        self._config_hash = hashlib.md5(config_string.encode('utf-8')).hexdigest()[:10] # Kürzerer Hash ist oft ausreichend
        logging.info(f"Generierter Konfigurations-Hash für Versionierung: {self._config_hash}")

        # 5. Validiere Split-Einstellungen
        if self.load_split and not self.split_definition_path:
            raise ValueError("Wenn 'load_split' gesetzt ist, muss auch 'split_definition_path' angegeben werden.")
        if self.load_split and self.load_split not in ['train', 'val', 'test']:
            logging.warning(f"Der angegebene Split-Name '{self.load_split}' ist unüblich. "
                            f"Üblich sind 'train', 'val', 'test'. Stelle sicher, dass der Schlüssel in der Split-Datei existiert.")

        # 6. Stelle sicher, dass RDKit verfügbar ist, wenn benötigt
        if not RDKIT_AVAILABLE:
             # Dieser Fehler sollte kritisch sein, da ohne RDKit nichts funktioniert
            logging.error("RDKit ist nicht installiert oder konnte nicht importiert werden. "
                          "Die Verarbeitung von Molekülen ist nicht möglich.")
            raise ImportError("RDKit wird benötigt, konnte aber nicht importiert werden.")

        # 7. Warnung bei num_workers != 1 (da Code auf sequentiell ausgelegt ist)
        if self.num_workers != 1:
            logging.warning(f"num_workers ist auf {self.num_workers} gesetzt, aber dieser Code ist primär für "
                            f"SEQUENTIELLE Verarbeitung (num_workers=1) optimiert und getestet. Setze auf 1.")
            self.num_workers = 1

        logging.debug("QM9Config __post_init__ abgeschlossen.")


# ==============================================================================
# Block 5: Kernfunktion: RDKit Mol -> PyG Data Konvertierung
# ==============================================================================
def rdkit_mol_to_pyg_data_configured(mol: Chem.Mol, config: QM9Config) -> Optional[Data]:
    """
    Konvertiert ein RDKit Molekül-Objekt in ein PyTorch Geometric Data-Objekt.

    Verwendet die Einstellungen aus dem übergebenen QM9Config-Objekt, um
    Features zu extrahieren und zu validieren. Verwendet limitierte Protokollierung
    für häufige Fehler.

    Args:
        mol (Chem.Mol): Das zu konvertierende RDKit Molekül-Objekt.
                        Es wird erwartet, dass das Molekül bereits eine 3D-Konformation hat
                        und die Zielwerte als Properties gespeichert sind.
        config (QM9Config): Das Konfigurationsobjekt mit den Verarbeitungseinstellungen.

    Returns:
        Optional[Data]: Ein PyG Data-Objekt bei Erfolg, sonst None bei Fehlern oder
                       wenn das Molekül übersprungen werden soll.
    """
    canonical_smiles = "[SMILES konnte nicht generiert werden]" # Platzhalter für Fehlerfälle
    try:
        # --- Vorprüfungen ---
        if mol is None:
            # Dies sollte selten passieren, wenn der SDMolSupplier korrekt arbeitet
            log_limited_warning("Überspringe 'None'-Molekül.", key="none_mol_input")
            return None
        if mol.GetNumAtoms() == 0:
            # Moleküle ohne Atome können nicht verarbeitet werden
            log_limited_warning("Überspringe Molekül ohne Atome.", key="zero_atoms")
            return None

        # Versuche, den kanonischen SMILES für Logging-Zwecke zu erhalten
        try:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception as smiles_error:
            log_limited_warning(f"Konnte SMILES nicht generieren: {smiles_error}", key="smiles_generation_error")
            # Fahre fort, wenn möglich, aber logge den Fehler

        num_atoms = mol.GetNumAtoms()

        # --- 1. Atompositionen (Konformation) ---
        try:
            conformer = mol.GetConformer() # Standardmäßig die erste Konformation
            positions = conformer.GetPositions() # Numpy-Array der 3D-Koordinaten
            pos_tensor = torch.tensor(positions, dtype=torch.float)
            # Sicherheitscheck: Stimmt die Anzahl der Positionen mit der Atomzahl überein?
            if pos_tensor.shape[0] != num_atoms:
                log_limited_warning(f"{canonical_smiles}: Anzahl Atompositionen ({pos_tensor.shape[0]}) != Atomzahl ({num_atoms}). Überspringe.",
                                    key="pos_atom_mismatch")
                return None
        except ValueError as conf_error: # Tritt auf, wenn keine Konformation vorhanden ist
            log_limited_warning(f"{canonical_smiles}: Keine 3D-Konformation gefunden. Überspringe. Fehler: {conf_error}",
                                key="conformer_missing")
            return None
        except Exception as e: # Andere unerwartete Fehler bei Konformationszugriff
            log_limited_warning(f"{canonical_smiles}: Fehler beim Zugriff auf Konformation: {e}. Überspringe.",
                                key="conformer_access_error")
            return None

        # --- 2. Atom-Features (Knoten-Features `x`) ---
        atom_features_list = [] # Liste zum Sammeln der Feature-Tensoren für jedes Atom
        atomic_numbers = []     # Liste der Ordnungszahlen (für spätere Referenz)

        for i, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            atomic_numbers.append(atom.GetAtomicNum())

            # Überprüfe, ob das Atomsymbol erlaubt ist
            if atom_symbol not in config._atom_symbol_map:
                log_limited_warning(f"{canonical_smiles}: Enthält unerlaubtes Atom '{atom_symbol}'. Überspringe.",
                                    key=f"disallowed_atom_{atom_symbol}")
                return None # Ganze Molekül verwerfen

            # a) One-Hot-Encoding des Atomsymbols (Basis-Feature)
            atom_type_onehot = torch.zeros(len(config.allowed_atom_symbols), dtype=torch.float)
            atom_type_onehot[config._atom_symbol_map[atom_symbol]] = 1.0
            current_atom_features = [atom_type_onehot]

            # b) Optionale Atom-Features hinzufügen (gemäß Konfiguration)
            if config.add_atomic_mass:
                current_atom_features.append(torch.tensor([atom.GetMass()], dtype=torch.float))
            if config.add_formal_charge:
                current_atom_features.append(torch.tensor([atom.GetFormalCharge()], dtype=torch.float))
            if config.add_hybridization:
                hybridization = atom.GetHybridization()
                # Mapping von RDKit Hybridisierungstypen zu Indizes für One-Hot
                hybridization_map = {
                    Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1,
                    Chem.HybridizationType.SP3: 2, Chem.HybridizationType.SP3D: 3,
                    Chem.HybridizationType.SP3D2: 4, Chem.HybridizationType.UNSPECIFIED: 5, # Unspezifiziert hinzufügen
                    Chem.HybridizationType.OTHER: 6 # Sonstige hinzufügen
                }
                num_hybridization_types = 7 # Anzahl der möglichen Typen im Mapping
                hybridization_onehot = torch.zeros(num_hybridization_types, dtype=torch.float)
                h_idx = hybridization_map.get(hybridization, 5) # Default auf UNSPECIFIED
                hybridization_onehot[h_idx] = 1.0
                current_atom_features.append(hybridization_onehot)
            if config.add_is_aromatic_atom:
                current_atom_features.append(torch.tensor([1.0 if atom.GetIsAromatic() else 0.0], dtype=torch.float))
            if config.add_is_in_ring_atom:
                current_atom_features.append(torch.tensor([1.0 if atom.IsInRing() else 0.0], dtype=torch.float))

            # Kombiniere alle Features für dieses Atom
            atom_features_list.append(torch.cat(current_atom_features))

        # Staple die Feature-Listen zu einem Tensor [num_atoms, num_node_features]
        x_tensor = torch.stack(atom_features_list, dim=0)

        # --- 3. Bindungs-Features (Kanten-Features `edge_attr`) und Kanten-Indizes (`edge_index`) ---
        edge_indices_list = []  # Liste von Tupeln (Startatom_idx, Endatom_idx)
        edge_features_list = [] # Liste der Feature-Tensoren für jede Kante

        if num_atoms > 1: # Bindungen nur möglich bei mehr als einem Atom
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()

                # Überprüfe, ob der Bindungstyp erlaubt ist
                if bond_type not in config._bond_type_map:
                    log_limited_warning(f"{canonical_smiles}: Enthält unerlaubten Bindungstyp '{bond_type}'. Überspringe.",
                                        key=f"disallowed_bond_{BOND_TYPE_TO_NAME.get(bond_type, str(bond_type))}")
                    return None # Ganze Molekül verwerfen

                # a) One-Hot-Encoding des Bindungstyps (Basis-Feature)
                bond_type_onehot = torch.zeros(len(config._allowed_bond_types_internal), dtype=torch.float)
                bond_type_onehot[config._bond_type_map[bond_type]] = 1.0
                current_bond_features = [bond_type_onehot]

                # b) Optionale Bindungs-Features hinzufügen (gemäß Konfiguration)
                if config.add_is_conjugated_bond:
                    current_bond_features.append(torch.tensor([1.0 if bond.GetIsConjugated() else 0.0], dtype=torch.float))
                if config.add_is_in_ring_bond:
                    current_bond_features.append(torch.tensor([1.0 if bond.IsInRing() else 0.0], dtype=torch.float))

                # Kombiniere alle Features für diese Bindung
                bond_feature_vector = torch.cat(current_bond_features)

                # Füge Kante in beide Richtungen hinzu (für ungerichteten Graphen)
                start_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices_list.extend([(start_idx, end_idx), (end_idx, start_idx)])
                edge_features_list.extend([bond_feature_vector, bond_feature_vector]) # Gleiche Features für beide Richtungen

        # Konvertiere Listen in Tensoren
        if edge_indices_list:
            edge_index = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous() # Shape [2, num_edges]
            edge_attr = torch.stack(edge_features_list, dim=0) # Shape [num_edges, num_edge_features]
        else:
            # Behandle den Fall ohne Kanten (z.B. einzelnes Atom)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            # Berechne die erwartete Dimensionalität der Kantenfeatures
            expected_edge_dim = len(config._allowed_bond_types_internal)
            if config.add_is_conjugated_bond: expected_edge_dim += 1
            if config.add_is_in_ring_bond: expected_edge_dim += 1
            edge_attr = torch.empty((0, expected_edge_dim), dtype=torch.float)

        # --- 4. Zielwerte (Targets `y`) ---
        target_values = []
        mol_properties = mol.GetPropsAsDict() # Alle im SDF gespeicherten Eigenschaften

        for target_key in config._actual_target_keys: # Iteriere über die *konfigurierten* Targets
            if target_key not in mol_properties:
                # Sollte nicht passieren, wenn die SDF-Datei vollständig ist
                log_limited_warning(f"{canonical_smiles}: Fehlender Zielwert '{target_key}' in Molekül-Properties. Überspringe.",
                                    key=f"missing_target_{target_key}")
                return None
            try:
                # Konvertiere den Wert zu float
                target_value = float(mol_properties[target_key])
                target_values.append(target_value)
            except (ValueError, TypeError) as convert_error:
                # Fehler bei der Konvertierung (z.B. wenn Wert kein String einer Zahl ist)
                log_limited_warning(f"{canonical_smiles}: Konnte Zielwert '{target_key}'='{mol_properties[target_key]}' nicht in Float konvertieren. Überspringe. Fehler: {convert_error}",
                                    key=f"convert_target_error_{target_key}")
                return None

        # Konvertiere Liste von Zielwerten in einen Tensor [1, num_targets]
        y_tensor = torch.tensor(target_values, dtype=torch.float).unsqueeze(0)

        # --- 5. Erzeuge PyG Data-Objekt ---
        data = Data(
            x=x_tensor,                # Knoten-Features [num_atoms, num_node_features]
            edge_index=edge_index,     # Kanten-Indizes [2, num_edges]
            edge_attr=edge_attr,       # Kanten-Features [num_edges, num_edge_features]
            pos=pos_tensor,            # Atompositionen [num_atoms, 3]
            y=y_tensor,                # Zielwerte [1, num_targets]
            smiles=canonical_smiles,   # Kanonischer SMILES-String (für Identifikation/Debugging)
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long), # Ordnungszahlen [num_atoms]
            num_nodes=num_atoms        # Anzahl der Knoten (redundant, aber nützlich)
        )

        # --- 6. Validierung (Optional, aber empfohlen) ---
        # `validate` prüft interne Konsistenz (z.B. ob edge_index auf gültige Knoten verweist)
        validation_result = data.validate(raise_on_error=False) # Nicht sofort abbrechen bei Fehler
        if not validation_result:
            # `validate` gibt bei Fehler False zurück und speichert Details in `data.validation_errors`
            error_details = getattr(data, 'validation_errors', 'Keine Details verfügbar')
            log_limited_warning(f"{canonical_smiles}: Interner Validierungsfehler im PyG Data-Objekt. Überspringe. Details: {error_details}",
                                key="pyg_validation_error")
            return None

        # Erfolgreich verarbeitet
        return data

    except Exception as e:
        # Fange alle unerwarteten Fehler während der Verarbeitung dieses Moleküls ab
        error_type_name = type(e).__name__
        log_key = f"unexpected_mol_proc_error_{error_type_name}"
        warning_counters[log_key] += 1 # Nutze den allgemeinen Zähler hier
        if warning_counters[log_key] <= 1: # Nur die erste Meldung dieses Fehlertyps vollständig loggen
            logging.error(f"Unerwarteter Fehler bei der Verarbeitung von Molekül (SMILES: {canonical_smiles}): {e}")
            logging.debug(traceback.format_exc()) # Detaillierter Traceback im Debug-Modus
        elif warning_counters[log_key] == 2:
            logging.error(f"Weitere unerwartete Fehler des Typs '{error_type_name}' werden unterdrückt...")
        return None # Überspringe dieses Molekül im Fehlerfall


# ==============================================================================
# Block 6: Hauptklasse für den QM9 InMemoryDataset
# ==============================================================================
class QM9EnhancedDataset(InMemoryDataset):
    """Erweiterte Klasse zum Laden und Verarbeiten des QM9-Datensatzes als PyG InMemoryDataset.

    Implementiert Download, sequentielle Verarbeitung von SDF zu PyG Data-Objekten
    unter Verwendung einer Konfiguration und optionales Laden von vordefinierten Splits.
    Speichert verarbeitete Daten und Metadaten.

    Args:
        config (QM9Config): Das Konfigurationsobjekt, das alle Einstellungen steuert.
        transform (Optional[Callable[[Data], Data]]): Eine Funktion/Transformation, die auf jedes Datenobjekt
                                        angewendet wird, wenn es abgerufen wird (dynamisch).
        pre_transform (Optional[Callable[[Data], Data]]): Eine Funktion/Transformation, die auf jedes Datenobjekt
                                            angewendet wird, bevor es auf der Festplatte gespeichert wird.
        pre_filter (Optional[Callable[[Data], bool]]): Eine Funktion, die auf jedes Datenobjekt angewendet wird,
                                         bevor es gespeichert wird. Wenn sie False zurückgibt,
                                         wird das Objekt verworfen.
    """
    def __init__(self,
                 config: QM9Config,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        # Überprüfe, ob RDKit verfügbar ist, bevor irgendetwas anderes getan wird
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit ist für QM9EnhancedDataset erforderlich, konnte aber nicht importiert werden.")

        self.config = config
        # Reset global counter for the main process duplicate check (if needed)
        self._main_process_duplicate_counter = 0
        logging.info(f"Initializing QM9EnhancedDataset in root: {self.config.root}...")

        # --- CRITICAL CHANGE: Define _raw_sdf_path *before* super().__init__ ---
        # This is needed because super().__init__ might call download(), which uses this path.
        # We use the raw_dir property here, which is safe as it computes the path on access.
        self._raw_sdf_path = osp.join(self.raw_dir, self.config.raw_sdf_name)
        # --- End Critical Change ---

        # Call super().__init__ AFTER defining attributes needed by its potential internal calls (like download/process)
        super().__init__(self.config.root, transform, pre_transform, pre_filter)

        # Now define other attributes that might depend on the superclass initialization or are not needed by it.
        self._processed_dir_path = osp.join(self.processed_dir, self.config.version_name) # Use processed_dir property

        # Interner Zähler für Meldungen im Hauptprozess (z.B. Duplikate)
        self._main_process_log_counters = defaultdict(int)

        logging.info(f"Root-Verzeichnis: {self.config.root}")
        logging.info(f"Verarbeitete Daten Version: {self.config.version_name}")
        logging.info(f"Verwendeter Konfigurations-Hash: {self.config._config_hash}")

        # Nach super().__init__ sollten die Daten geladen sein (entweder aus Datei oder durch process())
        # The base class __init__ already handles loading/processing.
        # We just need to check if data was loaded and apply splits if necessary.
        if not self._data_list and not (osp.exists(self.processed_paths[0]) or self.config.force_process):
             # This condition means super().__init__ didn't load or process data, which is unexpected
             # unless the processed file exists but is invalid, or some other edge case.
             # Let's rely on the base class logic. If it failed, it should have raised an error.
             # If it succeeded, self.data and self.slices should be populated by the base class.
             # We might need to explicitly load if the base class doesn't always do it.
             # Let's assume the base class loads it if found.
             pass # Base class handles loading or calls process()

        # Check if data is loaded after super().__init__
        if self.data is None:
             # This indicates an issue, either process() failed or loading failed silently in super
             logging.error(f"FEHLER: Daten wurden nach super().__init__() nicht geladen. Überprüfe process() Logs.")
             # Attempt to load manually again just in case, though this might hide underlying issues
             try:
                 self.data, self.slices = torch.load(self.processed_paths[0])
                 logging.info(f"Datensatz manuell nachgeladen aus: {self.processed_paths[0]}")
             except FileNotFoundError:
                 logging.error(f"FEHLER: Verarbeitete Datei wurde auch beim manuellen Versuch nicht gefunden: {self.processed_paths[0]}")
                 raise RuntimeError("Verarbeitete Daten konnten nicht geladen werden.")
             except Exception as e:
                 logging.error(f"FEHLER: Unerwarteter Fehler beim manuellen Nachladen der verarbeiteten Datei: {e}")
                 logging.debug(traceback.format_exc())
                 raise RuntimeError("Verarbeitete Daten konnten nicht geladen werden.") from e
        else:
             logging.info(f"Datensatz erfolgreich durch super().__init__ geladen/verarbeitet.")


        # Wenn ein spezifischer Split geladen werden soll, filtere die Daten jetzt
        if self.config.load_split:
            self._apply_split()

        logging.info(f"Initialisierung von QM9EnhancedDataset abgeschlossen. Datensatzgröße: {len(self)}")

    # --- Properties für Pfade (von InMemoryDataset gefordert) ---

    @property
    def raw_dir(self) -> str:
        """Gibt das Verzeichnis zurück, in dem Rohdaten gespeichert sind/werden."""
        # Liegt normalerweise direkt im root-Verzeichnis
        return osp.join(self.config.root, 'raw')

    @property
    def processed_dir(self) -> str:
        """
        Gibt das spezifische Verzeichnis für die *aktuelle Version* der verarbeiteten Daten zurück.
        Erstellt das Verzeichnis, falls es nicht existiert.
        """
        versioned_processed_path = osp.join(self.config.root, 'processed', self.config.version_name)
        os.makedirs(versioned_processed_path, exist_ok=True)
        # Die Basisklasse erwartet den *Basis*-Pfad des Verzeichnisses für verarbeitete Daten
        return osp.join(self.config.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        """
        Gibt eine Liste der Rohdateinamen zurück, die im `raw_dir` erwartet werden.
        Wird von `download()` verwendet, um zu prüfen, ob die Dateien existieren.
        """
        return [self.config.raw_sdf_name]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Gibt eine Liste der Dateinamen zurück, die im `processed_dir` nach der Verarbeitung erwartet werden.
        Der Dateiname enthält den Konfigurations-Hash, um verschiedene Verarbeitungsversionen zu ermöglichen.
        """
        return [f'qm9_pyg_data_{self.config._config_hash}.pt']

    # --- Methoden für Download und Verarbeitung (von InMemoryDataset aufgerufen) ---

    def download(self):
        """
        Lädt die QM9 Rohdaten (gdb9.tar.gz) herunter und extrahiert die SDF-Datei.
        Wird automatisch von `__init__` aufgerufen, wenn die Rohdaten nicht existieren.
        """
        # Stelle sicher, dass das raw-Verzeichnis existiert
        os.makedirs(self.raw_dir, exist_ok=True)
        logging.info(f"Prüfe auf Rohdaten in: {self.raw_dir}")

        # Pfad zur Zieldatei (SDF)
        target_sdf_path = self._raw_sdf_path
        # Pfad zur heruntergeladenen Archivdatei
        archive_path = osp.join(self.raw_dir, self.config.raw_archive_name)

        # Prüfen, ob die SDF-Datei bereits existiert
        if osp.exists(target_sdf_path) and not self.config.force_download:
            logging.info(f"Rohdatei '{self.config.raw_sdf_name}' existiert bereits. Download übersprungen.")
            return # Nichts zu tun

        # Wenn force_download gesetzt ist oder die Datei nicht existiert
        if osp.exists(target_sdf_path) and self.config.force_download:
            logging.warning(f"'force_download' ist gesetzt. Entferne existierende Rohdatei: {target_sdf_path}")
            try:
                os.remove(target_sdf_path)
            except OSError as e:
                logging.warning(f"Konnte existierende Rohdatei nicht entfernen: {e}")
        elif not osp.exists(target_sdf_path):
             logging.info(f"Rohdatei '{self.config.raw_sdf_name}' nicht gefunden. Starte Download...")

        # --- Download-Logik ---
        logging.info(f"Starte Download von: {self.config.qm9_url}")
        logging.info(f"Ziel-Archiv: {archive_path}")
        try:
            # Verwende stream=True für große Dateien und Fortschrittsanzeige
            response = requests.get(self.config.qm9_url, stream=True, timeout=120) # Timeout erhöht
            response.raise_for_status() # Fehler bei HTTP-Statuscodes >= 400

            # Gesamtgröße für tqdm (falls verfügbar)
            total_size = int(response.headers.get('content-length', 0))

            # Fortschrittsbalken initialisieren (oder Standard-Iterator verwenden)
            progress_bar = tqdm(
                total=total_size, unit='B', unit_scale=True,
                desc=f"Download {self.config.raw_archive_name}",
                disable=not TQDM_AVAILABLE, # Deaktivieren, wenn tqdm fehlt
                leave=False # Balken nach Abschluss entfernen
            )

            # Datei schreiben mit Fortschrittsanzeige
            with open(archive_path, 'wb') as f, progress_bar:
                for chunk in response.iter_content(chunk_size=8192): # Größere Chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk)) # Fortschritt aktualisieren

            # Überprüfen, ob die heruntergeladene Größe mit der erwarteten übereinstimmt
            if total_size != 0 and progress_bar.n != total_size:
                 raise IOError(f"Download unvollständig: Erwartet {total_size} Bytes, erhalten {progress_bar.n} Bytes.")

            logging.info("Download erfolgreich abgeschlossen.")

            # --- Extraktions-Logik ---
            logging.info(f"Extrahiere '{self.config.raw_sdf_name}' aus '{archive_path}' nach '{self.raw_dir}'...")
            extracted = False
            with tarfile.open(archive_path, 'r:gz') as tar: # Modus 'r:gz' für gzip-komprimierte tar-Archive
                # Finde das Mitglied (Datei) im Archiv, das auf .sdf endet (flexibler)
                sdf_member = next((m for m in tar.getmembers() if m.name.endswith(self.config.raw_sdf_name)), None)
                if sdf_member is None:
                    raise FileNotFoundError(f"SDF-Datei '{self.config.raw_sdf_name}' nicht im heruntergeladenen Archiv gefunden.")

                # Ändere den Namen des Mitglieds, um Pfadkomponenten zu entfernen (Sicherheit)
                sdf_member.name = osp.basename(sdf_member.name)
                tar.extract(sdf_member, path=self.raw_dir) # Extrahiere nur die SDF-Datei
                extracted = True

            if not extracted or not osp.exists(target_sdf_path):
                raise FileNotFoundError("Extraktion der SDF-Datei scheint fehlgeschlagen zu sein.")

            logging.info(f"Extraktion erfolgreich: '{target_sdf_path}' erstellt.")

            # --- Aufräumen: Archivdatei entfernen ---
            logging.info(f"Entferne heruntergeladene Archivdatei: {archive_path}")
            try:
                os.remove(archive_path)
                logging.info("Archivdatei erfolgreich entfernt.")
            except OSError as e:
                logging.warning(f"Konnte Archivdatei nach Extraktion nicht entfernen: {e}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Download fehlgeschlagen: Netzwerkfehler oder Serverproblem: {e}")
            logging.error("Bitte überprüfe deine Internetverbindung und die URL.")
            self._cleanup_failed_download(archive_path, target_sdf_path)
            raise ConnectionError(f"Download von {self.config.qm9_url} fehlgeschlagen.") from e
        except tarfile.TarError as e:
            logging.error(f"Extraktion fehlgeschlagen: Archivdatei ist möglicherweise beschädigt: {e}")
            self._cleanup_failed_download(archive_path, target_sdf_path)
            raise RuntimeError("Fehler beim Entpacken des Archivs.") from e
        except FileNotFoundError as e:
             logging.error(f"Extraktion fehlgeschlagen: {e}")
             self._cleanup_failed_download(archive_path, target_sdf_path)
             raise
        except Exception as e: # Fange andere unerwartete Fehler ab
            logging.error(f"Unerwarteter Fehler während Download/Extraktion: {e}")
            logging.debug(traceback.format_exc())
            self._cleanup_failed_download(archive_path, target_sdf_path)
            raise RuntimeError("Ein unerwarteter Fehler ist beim Download/Extraktion aufgetreten.") from e

    def _cleanup_failed_download(self, archive_path: str, sdf_path: str):
        """Hilfsfunktion zum Aufräumen nach fehlgeschlagenem Download/Extraktion."""
        logging.warning("Versuche, unvollständige Dateien zu bereinigen...")
        if osp.exists(archive_path):
            try:
                os.remove(archive_path)
                logging.info(f"Bereinigt: {archive_path}")
            except OSError as e:
                logging.warning(f"Bereinigung fehlgeschlagen für {archive_path}: {e}")
        if osp.exists(sdf_path):
             try:
                 os.remove(sdf_path)
                 logging.info(f"Bereinigt: {sdf_path}")
             except OSError as e:
                 logging.warning(f"Bereinigung fehlgeschlagen für {sdf_path}: {e}")

    def process(self):
        """
        Verarbeitet die rohe SDF-Datei SEQUENTIELL und erstellt die PyG Data-Objekte.

        Diese Methode wird automatisch von `__init__` aufgerufen, wenn die
        verarbeiteten Dateien (`processed_file_names`) nicht existieren oder
        `force_process=True` ist.
        """
        logging.info("=" * 60)
        logging.info(" Starte SEQUENTIELLE Verarbeitung der QM9 Rohdaten ".center(60, "="))
        logging.info("=" * 60)
        global warning_counters
        warning_counters = defaultdict(int) # Zähler für Warnungen zurücksetzen
        self._main_process_log_counters = defaultdict(int) # Zähler für Hauptprozess zurücksetzen
        start_time = time.time()

        # --- 1. Lese Moleküle aus der SDF-Datei ---
        logging.info(f"Lese Moleküle aus SDF-Datei: {self._raw_sdf_path}")
        logging.info(f"Einstellungen: {'Mit' if self.config.use_h_atoms else 'Ohne'} Wasserstoffatome, Sanitize=True")

        # Verwende einen SDMolSupplier. 'removeHs' steuert Wasserstoff, 'sanitize=True' führt RDKit-Bereinigungen durch.
        # Wichtig: `with` stellt sicher, dass die Datei korrekt geschlossen wird.
        try:
            # Zuerst zählen, um den Fortschrittsbalken zu initialisieren
            logging.info("Zähle Moleküleinträge in der SDF-Datei...")
            with Chem.SDMolSupplier(self._raw_sdf_path,
                                    removeHs=(not self.config.use_h_atoms),
                                    sanitize=True) as counter_supplier:
                if counter_supplier is None:
                    raise IOError(f"Konnte SDF-Datei zum Zählen nicht öffnen: {self._raw_sdf_path}")
                num_total_entries = len(counter_supplier) # Effizientes Zählen
            logging.info(f"{num_total_entries} Moleküleinträge gefunden.")
            if num_total_entries == 0:
                raise ValueError("Die SDF-Datei enthält keine Moleküleinträge. Verarbeitung abgebrochen.")

            # Jetzt die eigentliche Verarbeitung
            data_list: List[Data] = [] # Liste zum Sammeln der erfolgreichen Data-Objekte
            processed_count: int = 0
            skipped_total: int = 0
            duplicate_count: int = 0
            seen_smiles: Set[str] = set() # Set zum Verfolgen von Duplikaten

            logging.info(f"Verarbeite {num_total_entries} Moleküle sequentiell...")
            # Erneutes Öffnen des Suppliers für die Iteration
            with Chem.SDMolSupplier(self._raw_sdf_path,
                                    removeHs=(not self.config.use_h_atoms),
                                    sanitize=True) as supplier:
                if supplier is None:
                    raise IOError(f"Konnte SDF-Datei zur Verarbeitung nicht öffnen: {self._raw_sdf_path}")

                # Iteriere über Moleküle mit Fortschrittsanzeige
                mol_iterator = tqdm(supplier, total=num_total_entries, desc="SDF Verarbeitung (Sequentiell)", disable=not TQDM_AVAILABLE)
                for i, mol in enumerate(mol_iterator):
                    # --- a) Grundlegende Prüfung ---
                    if mol is None:
                        # Kann passieren, wenn RDKit ein Molekül nicht parsen kann
                        log_limited_warning(f"Eintrag {i+1}: Konnte Molekül nicht lesen/parsen. Überspringe.", key="mol_parsing_error")
                        skipped_total += 1
                        continue

                    # --- b) Konvertierung RDKit Mol -> PyG Data ---
                    # Diese Funktion enthält die detaillierte Logik und Fehlerbehandlung pro Molekül
                    pyg_data = rdkit_mol_to_pyg_data_configured(mol, self.config)

                    # --- c) Behandlung des Ergebnisses ---
                    if pyg_data is None:
                        # Fehler oder Filterung trat innerhalb der Konvertierungsfunktion auf
                        skipped_total += 1
                        continue # Zum nächsten Molekül

                    # --- d) Duplikatprüfung (optional) ---
                    if self.config.check_duplicates:
                        smiles = pyg_data.smiles
                        if smiles in seen_smiles:
                            self._main_process_log_counters['duplicate_skip'] += 1
                            if self._main_process_log_counters['duplicate_skip'] <= MAX_LOG_PER_WARNING_TYPE:
                                logging.debug(f"Duplikat übersprungen (SMILES): {smiles}")
                            elif self._main_process_log_counters['duplicate_skip'] == MAX_LOG_PER_WARNING_TYPE + 1:
                                logging.debug("Weitere Meldungen zu Duplikaten werden unterdrückt...")
                            duplicate_count += 1
                            skipped_total += 1
                            continue # Zum nächsten Molekül
                        else:
                            seen_smiles.add(smiles) # Neues Molekül, zum Set hinzufügen

                    # --- e) Pre-Filtering (optional) ---
                    data_is_valid = True
                    if self.pre_filter is not None and not self.pre_filter(pyg_data):
                        log_limited_warning(f"{getattr(pyg_data, 'smiles', '[SMILES N/A]')} durch pre_filter verworfen.", key="pre_filter_discard")
                        skipped_total += 1
                        data_is_valid = False
                        continue # Zum nächsten Molekül

                    # --- f) Pre-Transformation (optional) ---
                    if self.pre_transform is not None:
                        try:
                            transformed_data = self.pre_transform(pyg_data)
                            if transformed_data is None:
                                # Transformation hat None zurückgegeben, Molekül verwerfen
                                log_limited_warning(f"{getattr(pyg_data, 'smiles', '[SMILES N/A]')}: pre_transform gab None zurück. Überspringe.", key="pre_transform_returned_none")
                                skipped_total += 1
                                data_is_valid = False
                                continue # Zum nächsten Molekül
                            pyg_data = transformed_data # Verwende das transformierte Objekt
                        except Exception as transform_error:
                            # Fehler während der Transformation
                            log_key = f"pre_transform_error_{type(transform_error).__name__}"
                            warning_counters[log_key] += 1
                            if warning_counters[log_key] <= 1:
                                logging.error(f"Fehler bei pre_transform für {getattr(pyg_data, 'smiles', '[SMILES N/A]')}: {transform_error}")
                                logging.debug(traceback.format_exc())
                            elif warning_counters[log_key] == 2:
                                logging.error(f"Weitere Fehler des Typs '{type(transform_error).__name__}' bei pre_transform werden unterdrückt...")
                            skipped_total += 1
                            data_is_valid = False
                            continue # Zum nächsten Molekül

                    # --- g) Erfolgreich verarbeitet und alle Filter/Transformationen bestanden ---
                    data_list.append(pyg_data)
                    processed_count += 1

        except IOError as e:
            logging.error(f"Dateizugriffsfehler während der Verarbeitung der SDF-Datei: {e}")
            raise
        except ValueError as e: # Z.B. wenn SDF leer ist
            logging.error(f"Wertfehler während der Verarbeitung: {e}")
            raise
        except Exception as e:
            logging.error(f"Unerwarteter Fehler in der Hauptverarbeitungsschleife: {e}")
            logging.exception("Traceback:") # Loggt den vollständigen Traceback
            raise RuntimeError("Hauptverarbeitungsschleife fehlgeschlagen.") from e

        # --- 2. Verarbeitungszusammenfassung ---
        duration = time.time() - start_time
        logging.info("-" * 60)
        logging.info(" Verarbeitung abgeschlossen ".center(60, "-"))
        logging.info(f"Dauer: {duration:.2f} Sekunden")
        logging.info(f"Ursprüngliche Einträge: {num_total_entries}")
        logging.info(f"Erfolgreich verarbeitet: {processed_count}")
        if self.config.check_duplicates:
            logging.info(f"Übersprungene Duplikate: {duplicate_count}")
        other_skips = skipped_total - duplicate_count
        if other_skips > 0:
            logging.warning(f"Andere übersprungene Einträge (Fehler/Filterung): {other_skips}")
            logging.warning("Überprüfe frühere Protokolle (insb. WARNING/ERROR) für Details.")
        logging.info("-" * 60)

        # --- 3. WICHTIGE Prüfung: Ist die Liste leer? ---
        if not data_list:
            logging.error("FEHLER: Nach der Verarbeitung wurden keine gültigen Datenobjekte erstellt.")
            logging.error("Mögliche Ursachen: Probleme mit der SDF-Datei, zu strenge Filter, Fehler in der Konvertierung.")
            raise ValueError("Verarbeitung ergab einen leeren Datensatz. Abbruch.")

        # --- 4. Daten kollationieren und speichern ---
        # `collate` wird von InMemoryDataset bereitgestellt, um eine Liste von Data-Objekten
        # in ein großes Data-Objekt und dazugehörige Slices zu konvertieren.
        logging.info(f"Kollationiere {len(data_list)} erfolgreich verarbeitete Datenobjekte...")
        try:
            data, slices = self.collate(data_list)
            logging.info("Kollationierung erfolgreich.")
        except Exception as e:
            logging.error(f"Fehler beim Kollationieren der Daten: {e}")
            logging.debug(traceback.format_exc())
            raise RuntimeError("Datenkollationierung fehlgeschlagen.") from e

        # Pfad zur Zieldatei für die verarbeiteten Daten
        processed_data_path = self.processed_paths[0]
        logging.info(f"Speichere verarbeitete Daten und Slices nach: {processed_data_path}")

        try:
            # Stelle sicher, dass das Zielverzeichnis existiert (sollte durch self.processed_dir schon passiert sein)
            os.makedirs(osp.dirname(processed_data_path), exist_ok=True)
            # Speichere das kollationierte Datenobjekt und die Slices
            torch.save((data, slices), processed_data_path)
            logging.info("Speichern der verarbeiteten Daten abgeschlossen.")
        except Exception as e:
            logging.error(f"FEHLER beim Speichern der verarbeiteten Daten nach '{processed_data_path}': {e}")
            logging.debug(traceback.format_exc())
            # Versuche, die potenziell unvollständige Datei zu löschen
            if osp.exists(processed_data_path):
                try: os.remove(processed_data_path)
                except OSError: pass
            raise RuntimeError("Speichern der verarbeiteten Daten fehlgeschlagen.") from e

        # --- 5. Metadaten speichern ---
        try:
            self._save_metadata(processed_data_path, processed_count, duplicate_count, other_skips, duration)
        except Exception as e:
            # Fehler beim Speichern der Metadaten sollte die Verarbeitung nicht komplett fehlschlagen lassen
            logging.warning(f"Konnte Metadaten nicht speichern: {e}")

        logging.info("=" * 60)
        logging.info(" Verarbeitungsprozess erfolgreich beendet ".center(60, "="))
        logging.info("=" * 60)


    def _save_metadata(self, data_path: str, num_processed: int, num_duplicates: int, num_other_skips: int, duration: float):
        """
        Speichert Metadaten über den Verarbeitungslauf als JSON-Datei.

        Die Metadaten-Datei wird im selben Verzeichnis wie die verarbeitete Datendatei
        gespeichert, jedoch mit der Endung '.json'.

        Args:
            data_path (str): Pfad zur gespeicherten .pt-Datei.
            num_processed (int): Anzahl erfolgreich verarbeiteter Moleküle.
            num_duplicates (int): Anzahl übersprungener Duplikate.
            num_other_skips (int): Anzahl anderer übersprungener Moleküle.
            duration (float): Dauer des Verarbeitungsprozesses in Sekunden.
        """
        metadata_path = Path(data_path).with_suffix('.json')
        logging.info(f"Speichere Metadaten nach: {metadata_path}...")

        # Sammle Versionsinformationen
        try: import rdkit; rdkit_version = rdkit.__version__
        except ImportError: rdkit_version = "Nicht verfügbar"
        try: import torch_geometric; pyg_version = torch_geometric.__version__
        except ImportError: pyg_version = "Nicht verfügbar"
        try: import platform; py_version = platform.python_version()
        except Exception: py_version = "Nicht verfügbar"

        # Konfiguration serialisierbar machen (interne Felder entfernen)
        serializable_config = {}
        for f in fields(self.config):
            # Ignoriere interne Felder, die nicht Teil der ursprünglichen Konfiguration waren
            if not f.name.startswith('_'):
                value = getattr(self.config, f.name)
                # Konvertiere spezielle Typen (z.B. BondType Liste in Namen) falls nötig für JSON
                # Hier ist es meist schon serialisierbar, außer evtl. benutzerdefinierte Objekte
                serializable_config[f.name] = value

        # Metadaten-Dictionary erstellen
        metadata = {
            "processing_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "processing_duration_seconds": round(duration, 2),
            "qm9_config": serializable_config, # Füge die verwendete Konfiguration hinzu
            "dataset_statistics": {
                "successfully_processed": num_processed,
                "duplicates_skipped": num_duplicates,
                "other_skipped": num_other_skips,
                "total_in_sdf": num_processed + num_duplicates + num_other_skips, # Geschätzte Gesamtzahl
            },
            "environment_versions": {
                "python": py_version,
                "pytorch": torch.__version__,
                "torch_geometric": pyg_version,
                "rdkit": rdkit_version,
            },
            "output_files": {
                "processed_data_file": osp.basename(data_path),
                "metadata_file": osp.basename(metadata_path),
            },
            "source_data": {
                "sdf_filename": self.config.raw_sdf_name,
                "download_url": self.config.qm9_url,
            }
        }

        # Schreibe Metadaten als JSON-Datei
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                # `default=str` hilft bei der Serialisierung von Objekten, die JSON nicht direkt kennt
                json.dump(metadata, f, indent=4, default=str)
            logging.info("Metadaten erfolgreich gespeichert.")
        except TypeError as e:
            logging.error(f"Fehler bei der JSON-Serialisierung der Metadaten: {e}")
            logging.error("Möglicherweise enthält die Konfiguration nicht-serialisierbare Objekte.")
        except IOError as e:
            logging.error(f"Fehler beim Schreiben der Metadaten-Datei: {e}")
        except Exception as e:
            logging.error(f"Unerwarteter Fehler beim Speichern der Metadaten: {e}")


    def _apply_split(self):
        """
        Filtert die geladenen Daten basierend auf einer Split-Definitionsdatei.

        Diese Methode wird aufgerufen, wenn `config.load_split` und
        `config.split_definition_path` gesetzt sind, nachdem die Daten
        geladen wurden (entweder aus der Datei oder nach der Verarbeitung).
        """
        if not self.config.load_split or not self.config.split_definition_path:
            # Nichts zu tun, wenn kein Split geladen werden soll
            return

        split_file_path = Path(self.config.split_definition_path)
        split_name = self.config.load_split # z.B. 'train', 'val', 'test'
        split_key = f"{split_name}_idx" # Erwarteter Schlüssel in der Split-Datei

        logging.info(f"Wende Split '{split_name}' an...")
        logging.info(f"Lade Indizes aus Split-Definitionsdatei: {split_file_path}")

        if not split_file_path.is_file():
            raise FileNotFoundError(f"Split-Definitionsdatei nicht gefunden: {split_file_path}")

        try:
            # Lade die Split-Datei (erwartet ein Dictionary, z.B. {'train_idx': tensor, 'val_idx': tensor, ...})
            split_dict = torch.load(split_file_path)

            if not isinstance(split_dict, dict):
                raise TypeError(f"Split-Datei {split_file_path} enthält kein Dictionary.")

            # Hole die Indizes für den gewünschten Split
            indices = split_dict.get(split_key)
            if indices is None:
                raise ValueError(f"Schlüssel '{split_key}' nicht in Split-Datei {split_file_path} gefunden. "
                                 f"Verfügbare Schlüssel: {list(split_dict.keys())}")

            # Konvertiere Indizes sicher in einen LongTensor
            if not isinstance(indices, torch.Tensor):
                try:
                    # Versuche, aus Listen oder Numpy-Arrays zu konvertieren
                    indices = torch.tensor(indices, dtype=torch.long)
                except Exception as conv_err:
                    raise TypeError(f"Indizes für Split '{split_name}' in {split_file_path} konnten nicht "
                                    f"in einen Tensor konvertiert werden: {conv_err}") from conv_err
            if indices.dtype != torch.long:
                 logging.warning(f"Split-Indizes hatten Typ {indices.dtype}, konvertiere zu torch.long.")
                 indices = indices.to(torch.long)

            num_total_samples = len(self) # Anzahl der Samples *vor* dem Splitting
            num_split_indices = len(indices)

            logging.info(f"Anzahl der Indizes für Split '{split_name}': {num_split_indices} (aus insgesamt {num_total_samples} Samples)")

            if num_split_indices == 0:
                 logging.warning(f"Split '{split_name}' enthält keine Indizes. Der Datensatz wird leer sein.")
                 # Setze Daten auf leere Tensoren/Slices
                 self.data, self.slices = self.collate([]) # Erzeugt leere Struktur
            else:
                # Wichtige Prüfung: Sind die Indizes gültig?
                max_index = indices.max().item()
                if max_index >= num_total_samples:
                    raise IndexError(f"Maximaler Index ({max_index}) in Split '{split_name}' ist außerhalb des gültigen Bereichs "
                                     f"(Datensatz hat {num_total_samples} Samples, Indizes sollten 0 bis {num_total_samples-1} sein).")
                min_index = indices.min().item()
                if min_index < 0:
                    raise IndexError(f"Minimaler Index ({min_index}) in Split '{split_name}' ist negativ.")

                # Wende die Indizierung an, um nur die ausgewählten Datenpunkte zu behalten
                # `index_select` ist eine Methode von InMemoryDataset
                self.data, self.slices = self.index_select(indices)

                # Überprüfe die Größe nach dem Splitting
                final_size = len(self)
                logging.info(f"Datensatzgröße nach Anwendung von Split '{split_name}': {final_size}")
                if final_size != num_split_indices:
                    # Dies sollte normalerweise nicht passieren, wenn index_select korrekt funktioniert
                    logging.warning(f"Unerwartete Diskrepanz: Endgültige Größe ({final_size}) "
                                    f"entspricht nicht der Anzahl der Split-Indizes ({num_split_indices}).")

        except FileNotFoundError as e:
            logging.error(f"Fehler beim Anwenden des Splits: {e}")
            raise
        except (TypeError, ValueError, IndexError) as e:
            logging.error(f"Fehler beim Verarbeiten der Split-Datei oder Indizes: {e}")
            raise RuntimeError("Fehler beim Anwenden des Datensatz-Splits.") from e
        except Exception as e:
            logging.error(f"Unerwarteter Fehler beim Anwenden des Splits: {e}")
            logging.debug(traceback.format_exc())
            raise RuntimeError("Ein unerwarteter Fehler ist beim Anwenden des Splits aufgetreten.") from e


# ==============================================================================
# Block 7: Beispielhafte Ausführung und Demonstration
# ==============================================================================
if __name__ == '__main__':
    logging.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    logging.info(" Starte Beispielskript für QM9EnhancedDataset ".center(60, "~"))
    logging.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # --- Konfiguration für diesen Lauf ---
    # Verwende die korrigierten Target-Namen und sequentielle Verarbeitung.
    # Setze `force_process=False` (Standard), um nicht bei jedem Lauf neu zu prozessieren,
    # wenn die verarbeiteten Daten mit diesem Hash bereits existieren.
    run_config = QM9Config(
        root=osp.join(os.getcwd(), 'data', 'QM9_Seq_Demo'), # Separates Verzeichnis für diesen Demo-Lauf
        version_name="v1_demo_sequential",                 # Eigene Version für Demo
        target_keys_to_load=None,                          # Lade alle 12 korrigierten Standard-Targets
        # Deaktiviere optionale Features für einen schnellen Test
        add_formal_charge=False, add_is_aromatic_atom=False, add_hybridization=False,
        add_atomic_mass=False, add_is_conjugated_bond=False, add_is_in_ring_bond=False,
        check_duplicates=True,                             # Suche nach Duplikaten
        num_workers=1,                                     # Explizit sequentiell
        force_download=False,                              # Nicht neu herunterladen, wenn Rohdaten da sind
        force_process=False,                               # <--- KORRIGIERTER KOMMENTAR: Nicht neu verarbeiten, wenn .pt-Datei mit diesem Hash existiert.
        split_definition_path=None,                        # Kein Split in diesem Beispiel
        load_split=None
    )

    logging.info("Verwendete Konfiguration für diesen Lauf:")
    # Gib die Konfiguration übersichtlich aus
    config_dict = asdict(run_config)
    for key, value in config_dict.items():
         if not key.startswith('_'): # Interne Felder überspringen
             logging.info(f"  - {key}: {value}")
    logging.info("-" * 60)

    try:
        # Stelle sicher, dass das Root-Verzeichnis existiert
        os.makedirs(run_config.root, exist_ok=True)

        logging.info("Instanziiere QM9EnhancedDataset...")
        logging.info("(Dies löst ggf. Download und/oder Verarbeitung aus, falls nötig)")
        instantiation_start_time = time.time()

        # Globale Warnungszähler zurücksetzen, bevor die Verarbeitung beginnt
        warning_counters = defaultdict(int)

        # Erzeuge die Dataset-Instanz
        qm9_dataset = QM9EnhancedDataset(config=run_config)

        instantiation_end_time = time.time()
        logging.info(f"Instanziierung/Verarbeitung abgeschlossen.")
        logging.info(f"Benötigte Zeit: {instantiation_end_time - instantiation_start_time:.2f} Sekunden.")
        logging.info("-" * 60)

        # --- Inspektion des geladenen Datensatzes ---
        logging.info(" Datensatz Inspektion ".center(60, "="))

        logging.info(f"Typ des Datensatzobjekts: {type(qm9_dataset)}")
        logging.info(f"Anzahl der Graphen im Datensatz: {len(qm9_dataset)}")

        # Prüfe, ob der Datensatz leer ist (sollte nicht passieren, wenn alles gut lief)
        if len(qm9_dataset) == 0:
            logging.error("KRITISCH: Der Datensatz ist nach der Verarbeitung leer!")
            logging.error("Bitte überprüfe die Log-Ausgaben auf Fehler während der Verarbeitung.")
        else:
            # Zeige Details zum ersten Datenpunkt an
            logging.info("\n--- Details zum ersten Datenpunkt (Index 0) ---")
            first_data_point: Data = qm9_dataset[0]

            logging.info(f"Typ des Datenpunkts: {type(first_data_point)}")
            logging.info(f"Anzahl Knoten (Atome): {first_data_point.num_nodes}")
            logging.info(f"Anzahl Kanten (Bindungen*2): {first_data_point.num_edges}")
            logging.info(f"Form der Knoten-Features (x): {first_data_point.x.shape}")
            logging.info(f"Form der Kanten-Features (edge_attr): {first_data_point.edge_attr.shape}")
            logging.info(f"Form der Atompositionen (pos): {first_data_point.pos.shape}")
            logging.info(f"Form der Zielwerte (y): {first_data_point.y.shape}")
            logging.info(f"  -> Erwartet: [1, {len(run_config._actual_target_keys)}]")
            logging.info(f"Kanonischer SMILES: {first_data_point.smiles}")
            logging.info(f"Atom-Ordnungszahlen (atomic_numbers): {first_data_point.atomic_numbers.tolist()}")
            logging.info(f"Zielwerte (y): {first_data_point.y.tolist()[0]}") # Erste (und einzige) Zeile ausgeben

            # Zusätzliche Validierung der Target-Dimension
            assert first_data_point.y.shape[1] == len(run_config._actual_target_keys), \
                f"Fehler: Anzahl der Zielwerte ({first_data_point.y.shape[1]}) stimmt nicht mit der Konfiguration " \
                f"({len(run_config._actual_target_keys)}) überein!"

            logging.info("\n--- Details zum letzten Datenpunkt ---")
            last_data_point: Data = qm9_dataset[-1]
            logging.info(f"SMILES: {last_data_point.smiles}")
            logging.info(f"Anzahl Knoten: {last_data_point.num_nodes}")
            logging.info(f"Form y: {last_data_point.y.shape}")

            logging.info("\nGrundlegende Inspektion erfolgreich abgeschlossen.")

        logging.info("=" * 60)
        logging.info(" Beispielskript erfolgreich beendet ".center(60, "="))
        logging.info("=" * 60)

    # --- Fehlerbehandlung für das Hauptskript ---
    except FileNotFoundError as e:
        logging.error(f"\n!!! Dateifehler im Hauptskript: {e}")
        logging.exception("Traceback:")
    except ValueError as e:
        logging.error(f"\n!!! Wertfehler im Hauptskript: {e}")
        logging.exception("Traceback:")
    except RuntimeError as e:
        logging.error(f"\n!!! Laufzeitfehler im Hauptskript: {e}")
        logging.exception("Traceback:")
    except ConnectionError as e:
         logging.error(f"\n!!! Verbindungsfehler im Hauptskript (wahrscheinlich beim Download): {e}")
         logging.exception("Traceback:")
    except ImportError as e:
         logging.error(f"\n!!! Importfehler: {e}")
         logging.error("Stelle sicher, dass alle Abhängigkeiten (PyTorch, PyG, RDKit, requests, tqdm) installiert sind.")
         logging.exception("Traceback:")
    except AssertionError as e:
         logging.error(f"\n!!! Assertionsfehler im Hauptskript (interne Prüfung fehlgeschlagen): {e}")
         logging.exception("Traceback:")
    except IndexError as e:
         logging.error(f"\n!!! Indexfehler im Hauptskript (oft bei Splits oder Datenzugriff): {e}")
         logging.exception("Traceback:")
    except Exception as e: # Fängt alle anderen unerwarteten Fehler ab
        logging.error(f"\n!!! Unerwarteter Fehler im Hauptskript aufgetreten !!!")
        logging.error(f"Fehlertyp: {type(e).__name__}")
        logging.error(f"Details: {e}")
        logging.exception("Traceback:") # Gibt den vollständigen Traceback aus

    finally:
        logging.info("Skriptausführung beendet.")
