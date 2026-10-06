import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

# 1. Chemin vers votre fichier binaire .wandb
run_file = "wandb/offline-run-20260717_132800-xl536t5t/run-xl536t5t.wandb"

if not os.path.exists(run_file):
    raise FileNotFoundError(f"Fichier introuvable : {run_file}")

print(f"Extraction des métriques du fichier binaire : {run_file}...")

# 2. Lire et décoder le fichier binaire
ds = DataStore()
ds.open_for_scan(run_file)

history_data = []

while True:
    data = ds.scan_data()
    if data is None:
        break # Fin du fichier binaire
    
    try:
        # Initialise un objet Record Protobuf vide
        record = wandb_internal_pb2.Record()
        # Remplit l'objet avec les données binaires lues
        record.ParseFromString(data)
        
        # On ne s'intéresse qu'aux enregistrements de type 'history' (les logs d'entraînement)
        if record.WhichOneof('record_type') == 'history':
            row = {}
            for item in record.history.item:
                # Gère les clés imbriquées s'il y en a
                key = '/'.join(item.nested_key) if len(item.nested_key) > 0 else item.key
                row[key] = json.loads(item.value_json)
            if row:
                history_data.append(row)
    except Exception as e:
        continue

ds.close()

# 3. Convertir en DataFrame Pandas
if history_data:
    df = pd.DataFrame(history_data)
    print("\nExtraction réussie ! Métriques disponibles :")
    for col in df.columns:
        if not col.startswith("_"):
            print(f" - {col}")

    # 4. Générer les graphiques
    plt.figure(figsize=(12, 5))
    x_axis = 'epoch' if 'epoch' in df.columns else '_step'
    # x_axis = '_step'h

    # Graphique 1 : La Perte (Loss)
    plt.subplot(1, 2, 1)
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]
    if loss_cols:
        for col in loss_cols:
            clean_df = df[[x_axis, col]].dropna()
            plt.plot(clean_df[x_axis], clean_df[col], label=col)
        plt.title("Évolution de la Perte (Loss)")
        plt.xlabel("Époques" if x_axis == 'epoch' else "Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

    # Graphique 2 : Le Dice (Segmentation)
    plt.subplot(1, 2, 2)
    dice_cols = [c for c in df.columns if 'dice' in c.lower()]
    if dice_cols:
        for col in dice_cols:
            clean_df = df[[x_axis, col]].dropna()
            plt.plot(clean_df[x_axis], clean_df[col], label=col)
        plt.title("Évolution du Score Dice")
        plt.xlabel("Époques" if x_axis == 'epoch' else "Steps")
        plt.ylabel("Dice")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("wandb_metrics.png")
    print("\nGraphiques sauvegardés sous 'wandb_metrics.png'.")
    plt.show()
else:
    print("\nAucune donnée d'historique (history) n'a été trouvée dans le fichier.")
    print("Il est fort probable que le script d'entraînement ait crashé trop tôt (Époque 0) sans enregistrer de logs.")