# Single Cell Structure Analysis

Contains jupyter notebooks analysing single cell Hi-C structures. Relies on [NucFrames code](https://github.com/latkins/nuc_frames).

The file loop_analysis.ipynb performs monte carlo analysis of CTCF/Cohesin loops.

## Installation

Requires python3. Using Anaconda to create a fresh environment, and then manage
these requirements, is suggested. All packages can be installed with pip or
conda.

### Python Packages

* Seaborn
* pyliftover
* jupyter

### Running

Run ```jupyter notebook``` in the cloned folder. A web browser should open. View
```loop_analysis.ipynb``` to view and run the analysis.

### Data Files

The ```.hdf5``` files required to run these scripts are availiable from GEO,
accession number GSE80280. These files must be downloaded prior to running these
scripts.

Additionally, a file defining loop locations is required
\([https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525)\).
This file was generated
by [Rao et al.](https://www.ncbi.nlm.nih.gov/pubmed/25497547).
