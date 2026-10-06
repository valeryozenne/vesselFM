#!/usr/bin/env bash

# Permet de passer la partition en argument (ex: ./script.sh gpu_p10). 
# Par défaut, utilise gpu_p6.
PARTITION=${1:-gpu_p6}
SESSION_NAME="vesselFM"

# Récupère le JobID (%i), JobName (%j) et NodeList (%N) pour le job en cours (RUNNING)
JOB_INFO=$(squeue --me --states=R --partition="$PARTITION" -h -o "%i %j %N" | head -n 1)

if [ -z "$JOB_INFO" ]; then
  echo "Erreur : Aucun job en cours (RUNNING) trouvé sur la partition $PARTITION."
  exit 1
fi

# Assigne les valeurs aux variables
read JOB_ID JOB_NAME NODE_NAME <<< "$JOB_INFO"

echo "Job actif trouvé : $JOB_NAME (ID: $JOB_ID) sur le noeud $NODE_NAME"

# Construit le chemin des logs en fonction de %x (JOB_NAME) et %j (JOB_ID)
# Utilisation de tail -F (majuscule) pour patienter si le fichier n'est pas encore créé
CMD_ERR="tail -F logs/${JOB_NAME}-${JOB_ID}.err"
CMD_OUT="tail -F logs/${JOB_NAME}-${JOB_ID}.out"
CMD_SSH="watch ssh $NODE_NAME nvidia-smi"

# Vérifie si la session existe déjà pour éviter les doublons
tmux has-session -t "$SESSION_NAME" 2>/dev/null

if [ $? != 0 ]; then
  # 1. Crée une nouvelle session détachée (-d)
  tmux new-session -d -s "$SESSION_NAME"

  # 2. Découpe en 4 panneaux
  tmux split-window -v -t "$SESSION_NAME:0"       # Coupe haut / bas
  tmux split-window -h -t "$SESSION_NAME:0.1"     # Coupe le volet du bas (index 1)
  tmux split-window -h -t "$SESSION_NAME:0.0"     # Coupe le volet du haut (index 0)

  # 3. Répartit les 4 volets équitablement en grille (2x2)
  tmux select-layout -t "$SESSION_NAME:0" tiled

  # 4. Envoie les commandes dynamiques dans chaque panneau (index 0, 1, 2, 3)
  tmux send-keys -t "$SESSION_NAME:0.0" "watch squeue --me" C-m
  tmux send-keys -t "$SESSION_NAME:0.1" "$CMD_ERR" C-m
  tmux send-keys -t "$SESSION_NAME:0.2" "$CMD_SSH" C-m
  tmux send-keys -t "$SESSION_NAME:0.3" "$CMD_OUT" C-m

  # 5. Positionne le curseur sur le premier panneau
  tmux select-pane -t "$SESSION_NAME:0.0"
fi

# Attache le terminal à la session
tmux attach-session -t "$SESSION_NAME"