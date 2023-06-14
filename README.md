# Closest Points'
## project overview 
Implemented an algorithm in Python to efficiently choose sets of points to determine where to place telephone 
poles. This functionality was implemented using random algorithms such as simulated annealing and other Monte 
Carlo algorithms to efficiently solve an NP-hard problem.

## Requirements

A Python skeleton is available in the `python` subdirectory. The Python
skeleton was developed using Python 3.9, but it should work with Python
versions 3.6+.

## Usage

Run `solve_all.py` by executing the following command:
`python3.8 solve_all.py`


## How to run Our Solver:

To solve a instance run command within the python directory:  

python3 solve.py .\inputs\XXX\YYY.in --solver=anneal_1 .\outputs\XXX\YYY.out  

The output is placed in .\outputs\XXX\YYY.out  

To visualize:  

python3 .\visualize.py .\inputs\XXX\YYY.in --with-solution .\outputs\XXX\YYY.out out.svg  

(There is a VSCode extension to preview svg files easily in the IDE)  
    
### Generating instances

To generate instances, read through [`python/instance.py`](python/instance.py),
which contains a dataclass (struct) that holds the data for an instance, as
well as other relevant methods. Then modify the
[`python/generate.py`](python/generate.py) file by filling in the
`make_{small,medium,large}_instance` functions.

After you have filled in those functions, you can run `make generate` in the
`python` directory to generate instances into the input directory.

To run unit tests, run `make check`.

## Visualizing Instances

To visualize problem instances, run `python3 visualize.py`, passing  in the path to your 
`.in` file as the first argument (or `-` to read from standard input). To visualize a solution
as well, pass in a `.out` file to the option `--with-solution`.

By default, the output visualization will be written as a SVG file to standard output.
To redirect it to a file, use your shell's output redirection or pass in an output file as
an additional argument.

For example, you could run
```bash
python3 visualize.py my_input.in out.svg
```
to create an `out.svg` file visualizing the `my_input.in` problem instance.

To visualize a solution file for this instance as well, you could run
```bash
python3 visualize.py my_input.in --with-solution my_soln.out out.svg
```






