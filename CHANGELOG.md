# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-29

### Added

- Initial release of Polnet, a comprehensive tool for generating synthetic cryo-electron tomograms
- Core simulation capabilities for biological structures:
  - **Membranes**: Simulation of cellular membranes with spherical, ellipsoidal, and toroidal geometries
  - **Filaments**: Simulation of cytoskeletal structures including actin filaments and microtubules with helicoidal geometry
  - **Protein Complexes**: Simulation of globular protein clusters using Self-Avoiding Walk on Lattice (SAWLC) networks
  - **Membrane Proteins**: Simulation of membrane-bound protein complexes
- **Data Generation Pipeline**:
  - Configurable tomogram dimensions and voxel sizes
  - Adjustable occupancy and density parameters
  - Support for multiple biological structures in single tomograms
- **Output Formats**:
  - Ground truth density maps (`.mrc` format)
  - Segmentation label maps (`.mrc` format)  
  - 3D polydata models (`.vtp` format) for visualization
  - Simulated micrograph tilt-series for realistic cryo-ET data
- **TEM Simulation**:
  - Configurable tilt angles and detector parameters
  - Signal-to-noise ratio controls
  - Misalignment simulation for realistic data

### Technical Details

- Python-based implementation with scientific computing stack
- Support for Docker containerization
