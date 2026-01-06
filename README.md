# mom-for-MIMo

Adding a mom to the [MIMo](https://github.com/trieschlab/MIMo/tree/main) platform.

This repository contains the supplementay code for the masters thesis called Adding a Caregiver to a Baby Simulator (2026). The goal of the thesis was to expand the [MIMo](https://github.com/trieschlab/MIMo/tree/main) simulator by adding a caregiver model the infant can interact with.

## Structure
- `mom_mimoEnv` contains scenes, models of adults and infants and textures derived from [MIMo](https://github.com/trieschlab/MIMo).
- `motion_retargeting` contains files neccessary for applying movement exported from motion capture to the models in the simulation
- `xml_work` contains files for resizing the mother model from `mom_mimoEnv`
- `generate_updated_model_w_mocap_equalities.py` runs the resizing process from the previous folder and creates also all the necessary XML files needed for motion retargeting related to the resized model
- `run_motion_retargeting.py` runs the simulation scenarios and saves the selected outputs
## How to use the contents of this repository
1. Install the [BabyBench (2025)](https://babybench.github.io/2025/) MIMo environment, follow the official [installation guide](https://babybench.github.io/2025/installation/) 
2. Download this repository
3. To run the prepared `mom1` simulation, execute `python run_motion_retargeting.py`
4. For other interaction scenarios and simulation results, see the [our Google Drive](https://drive.google.com/drive/folders/1AQJQcR1kb0gsvE73_v49ukzLqOPVfkB7?usp=sharing)
