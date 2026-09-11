import sys
import os
import glob
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTreeWidget, QTreeWidgetItem,
                             QFileDialog, QMessageBox, QLabel)
from PyQt6.QtCore import Qt

class NiftiSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIfTI MRView Selector (PyQt6)")
        self.resize(900, 600)

        self.root_dir = ""
        self.out_dir = ""
        self.file_items = []  # Stocke les objets représentant les fichiers

        self.setup_ui()

    def setup_ui(self):
        # Widget principal et Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Barre d'outils ---
        toolbar = QHBoxLayout()
        
        self.btn_load = QPushButton("1. Charger la Racine")
        self.btn_load.clicked.connect(self.load_tree)
        toolbar.addWidget(self.btn_load)

        self.btn_out = QPushButton("2. Dossier de Sortie (Liens)")
        self.btn_out.clicked.connect(self.set_out_dir)
        toolbar.addWidget(self.btn_out)

        self.btn_symlink = QPushButton("3. Générer les Liens Symboliques")
        self.btn_symlink.clicked.connect(self.create_symlinks)
        self.btn_symlink.setStyleSheet("background-color: #c8e6c9;")
        toolbar.addWidget(self.btn_symlink)
        
        layout.addLayout(toolbar)

        # --- Instructions ---
        lbl = QLabel("Double-clic: Ouvrir l'image avec mrview | Cochez la case (ou Espace) pour sélectionner")
        lbl.setStyleSheet("color: gray; padding-top: 5px; padding-bottom: 5px;")
        layout.addWidget(lbl)

        # --- Arbre des fichiers ---
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Arborescence des fichiers .nii.gz")
        self.tree.itemDoubleClicked.connect(self.on_double_click)
        layout.addWidget(self.tree)

    def load_tree(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier racine")
        if not dir_path: 
            return
            
        self.root_dir = dir_path
        self.tree.clear()
        self.file_items.clear()

        # Recherche des fichiers (selon votre arborescence)
        pattern = os.path.join(self.root_dir, "*", "*CT*", "RAW-NIFTI", "*.nii.gz")
        files = glob.glob(pattern)

        if not files:
            QMessageBox.information(self, "Résultat", "Aucun fichier correspondant n'a été trouvé.")
            return

        # Construction du dictionnaire pour l'arborescence
        tree_dict = {}
        for f in files:
            rel = os.path.relpath(f, self.root_dir)
            parts = rel.split(os.sep)
            curr = tree_dict
            for part in parts:
                if part not in curr: 
                    curr[part] = {}
                curr = curr[part]

        # Remplissage de l'interface
        self.insert_nodes(self.tree.invisibleRootItem(), tree_dict, self.root_dir)
        self.tree.expandAll() # Déploie l'arbre par défaut

    def insert_nodes(self, parent_item, nodes_dict, current_path):
        for key, sub_dict in sorted(nodes_dict.items()):
            full_path = os.path.join(current_path, key)
            is_leaf = len(sub_dict) == 0

            item = QTreeWidgetItem(parent_item, [key])
            
            if is_leaf:
                # Stocker le chemin du fichier de façon invisible (PyQt6 utilise Qt.ItemDataRole)
                item.setData(0, Qt.ItemDataRole.UserRole, full_path) 
                
                # Ajouter la case à cocher (PyQt6 utilise Qt.ItemFlag et Qt.CheckState)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                
                self.file_items.append(item)
            else:
                self.insert_nodes(item, sub_dict, full_path)
    
    def on_double_click(self, item, column):
        # Récupérer le chemin caché (retourne None si ce n'est pas un fichier terminal)
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path: 
            try:
                subprocess.Popen(["mrview", path])
            except FileNotFoundError:
                QMessageBox.critical(self, "Erreur", "La commande 'mrview' n'est pas accessible. Lancez ce script depuis un terminal où MRtrix3 est chargé.")
    
    def set_out_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de destination")
        if dir_path:
            self.out_dir = dir_path
            QMessageBox.information(self, "Dossier défini", f"Les liens seront créés dans :\n{self.out_dir}")

    def create_symlinks(self):
        if not self.out_dir:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord choisir un dossier de sortie (Étape 2).")
            return
        
        # Récupérer les items dont la case est cochée (PyQt6 utilise Qt.CheckState.Checked)
        selected = [item for item in self.file_items if item.checkState(0) == Qt.CheckState.Checked]
        
        if not selected:
            QMessageBox.warning(self, "Attention", "Aucun fichier n'a été sélectionné.")
            return
        
        count = 0
        for item in selected:
            src = item.data(0, Qt.ItemDataRole.UserRole)
            parts = Path(src).parts
            
            # Nom unique : dossier_parent2_dossier_parent1_fichier.nii.gz
            if len(parts) >= 4:
                name = f"{parts[-4]}_{parts[-3]}_{parts[-1]}" 
            else:
                name = os.path.basename(src)
                
            dst = os.path.join(self.out_dir, name)
            
            try:
                # lexists détecte même un lien symbolique "cassé" pour pouvoir le remplacer
                if os.path.lexists(dst): 
                    os.remove(dst)
                os.symlink(src, dst)
                count += 1
            except Exception as e:
                print(f"Impossible de lier {src}: {e}")
        
        QMessageBox.information(self, "Succès", f"{count} liens symboliques ont été créés avec succès !")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NiftiSelectorApp()
    ex.show()
    # PyQt6 utilise exec() au lieu de exec_()
    sys.exit(app.exec())