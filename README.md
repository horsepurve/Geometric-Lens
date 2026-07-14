# Geometric Lens: joint visualization of LLM decision boundaries and reasoning trajectories

<div align="center">
<img src="figures/llm-decision-boundary.png" width="70%" alt="LLM Decision Boundary & Trajectory Visualization for 'The capital of France is'">
</div>
<div align="center">
<img src="figures/llm-reasoning-trajectory.png" width="70%" alt="LLM Decision Boundary & Trajectory Visualization for In-Context Interference">
</div>

### Installation
```bash
pip install numpy torch transformers
```

### Hidden representation readout and visualization
Run Logit Lens: 01_logit_lens_patchscopes.ipynb

Run Patchscopes: 01_logit_lens_patchscopes.ipynb

Run Geometric Lens (ours): 02_geometric_lens.ipynb

Compare all lenses in 2D: 03_planar visualization.ipynb

### References
```tex
@misc{ma2026laguerregeometryinterpretinglarge,
      title={Laguerre Geometry for Interpreting Large Language Models}, 
      author={Chunwei Ma and Russell Wolfinger},
      year={2026},
      eprint={2607.10578},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.10578}, 
}
```
