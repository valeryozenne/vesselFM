#!/bin/bash

source /workspace_QMRI/USERS_CODE/hsalles/00_venvs/vesselFM_env/bin/activate

# None geom, perlin noise, 20000 samples
python vesselfm/d_drand/gen_bg.py

# voronoi, perlin noise, 10000 samples
python vesselfm/d_drand/gen_bg.py --mode_bg_geom voronoi --mode_bg_noise perlin --num_imgs 10000

# spheres, perlin noise, 10000 samples
python vesselfm/d_drand/gen_bg.py --mode_bg_geom spheres --mode_bg_noise perlin --num_imgs 10000

# None geom, plain noise, 10000 samples
python vesselfm/d_drand/gen_bg.py --mode_bg_geom none --mode_bg_noise plain --num_imgs 10000

# merge all the generated backgrounds with forgrounds
python vesselfm/d_drand/gen_data.py