
# Nyx environment
/ptmp/bin/salloc-a-node.sh -g --jobname chris_dev_gpu --partition gpu --time 04:00:00
cd ~/cvar_generation
module load singularity
singularity shell --nv /ptmp/containers/cpilab-transformers_latest-2021-09-27-7fb926404b36.sif
bash
conda activate /home/cgagne/cvar_generation/conda_env

# environment installation
conda create -y -p ~/cvar_generation/conda_env
conda activate /home/cgagne/cvar_generation/conda_env
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c huggingface transformers
export TRANSFORMERS_CACHE=~/cvar_generation/cache
echo "$TRANSFORMERS_CACHE"
conda install -c huggingface -c conda-forge datasets
pip install ipdb
conda install scikit-learn
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
conda install -c conda-forge jupyter_contrib_nbextensions
conda install -c conda-forge sentence-transformers
conda install matplotlib
conda install seaborn

# other things installed but not necessary (for quantile dqn code)

pip install gym
pip install pygame
pip install tabulate
