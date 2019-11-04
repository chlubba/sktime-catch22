# sktime-catch22

This repository provides a random forest classifier, `catch22Forest`, for time-series based on [`catch22`](https://github.com/chlubba/catch22) features, a collection of 22 time-series features selected by their classification performance from a much larger set of 7500+ features of the [_hctsa_](https://github.com/benfulcher/hctsa.git) toolbox. Features are implemented in C and wrapped for Python. `catch22` is distributed under the GNU General Public License Version 3.

## Installation

### Dependencies

`catch22Forest` requires

 * Python (>= 3.4)
 * NumPy (>= 1.8.2)
 * sktime (>= 0.3.0)
 * catch22 (>=0.0.1)

### Compilation

Using `setuptools`

    python setup.py install
	
To install the requirements, use:

    pip install -r requirements.txt

## Usage

See the examples folder, containing two examples:
 * [Univariate time series](https://github.com/chlubba/sktime-catch22/examples/exampleUni.py)
 * [Integration with sktime](https://github.com/chlubba/sktime-catch22/examples/examplesktime.py)
	
## References

For information on how this feature set was constructed see the paper:

* C.H. Lubba, S.S. Sethi, P. Knaute, S.R. Schultz, B.D. Fulcher, N.S. Jones. [_catch22_: CAnonical Time-series CHaracteristics](https://doi.org/10.1007/s10618-019-00647-x). *Data Mining and Knowledge Discovery* **33**, 6 (2019).

For information on the full set of over 7000 features, see the following (open) publications:

* B.D. Fulcher and N.S. Jones. [_hctsa_: A computational framework for automated time-series phenotyping using massive feature extraction](http://www.cell.com/cell-systems/fulltext/S2405-4712\(17\)30438-6). *Cell Systems* **5**, 527 (2017).
* B.D. Fulcher, M.A. Little, N.S. Jones [Highly comparative time-series analysis: the empirical structure of time series and their methods](http://rsif.royalsocietypublishing.org/content/10/83/20130048.full). *J. Roy. Soc. Interface* **10**, 83 (2013).