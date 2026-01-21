# Data Exploration Notebooks

This directory contains Jupyter notebooks for exploring and understanding the datasets.

## Notebooks

### `explore_reasonaqa.ipynb`
Comprehensive exploration of the ReasonAQA dataset including:
- Dataset structure and statistics
- Task type distribution
- Audio file path analysis
- Single vs dual audio task comparison
- Input/answer length analysis
- Sample filtering and searching
- Visualizations

## Usage

1. **Activate your conda environment:**
   ```bash
   conda activate qwen_omni
   ```

2. **Install Jupyter (if not already installed):**
   ```bash
   pip install jupyter notebook matplotlib seaborn pandas
   ```

3. **Launch Jupyter:**
   ```bash
   cd /home/ikulkar1/qwen_omni_finetune/audio_reasoning_interspeech/notebooks
   jupyter notebook
   ```

4. **Open the notebook** `explore_reasonaqa.ipynb` in your browser

## Tips

- Run cells sequentially from top to bottom
- Use the last cells for your custom exploration
- DataFrames can be filtered and searched interactively
- Visualizations help understand data distributions
- Export subsets for quick testing

## Data Locations

- **ReasonAQA**: `../src/data/reasonaqa/reasonaqa/`
  - `train.json` - 968,071 samples
  - `val.json` - 114,188 samples
  - `test.json` - 161,695 samples

- **MMAR**: `../src/data/MMAR-meta.json`
  - 1,000 samples for audio reasoning evaluation
