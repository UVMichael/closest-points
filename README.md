# Spring 2022 CS170 Project - Team 'Second Rate Algorithm'

## How to run Our Solver:

To solve a instance run command within the python directory:  

python3 solve.py .\inputs\XXX\YYY.in --solver=anneal_1 .\outputs\XXX\YYY.out  

The output is placed in .\outputs\XXX\YYY.out  

To visualize:  

python3 .\visualize.py .\inputs\XXX\YYY.in --with-solution .\outputs\XXX\YYY.out out.svg  

(There is a VSCode extension to preview svg files easily in the IDE)  
    

    

## Requirements

A Python skeleton is available in the `python` subdirectory. The Python
skeleton was developed using Python 3.9, but it should work with Python
versions 3.6+.

## Usage

Run `solve_all.py` by executing the following command:
`python3.8 solve_all.py`

### Generating instances

To generate instances, read through [`python/instance.py`](python/instance.py),
which contains a dataclass (struct) that holds the data for an instance, as
well as other relevant methods. Then modify the
[`python/generate.py`](python/generate.py) file by filling in the
`make_{small,medium,large}_instance` functions.

After you have filled in those functions, you can run `make generate` in the
`python` directory to generate instances into the input directory.

To run unit tests, run `make check`.

## Solving

We've created a solver skeleton at [`python/solve.py`](python/solve.py).

The solver writes the solution to stdout. To write to a file, use your shell's
stdout redirection:

```
python3 solve.py case.in --solver=naive > case.out
```

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


## Other tooling

We may provide additional tooling, including a tool that calls a solver on all
inputs in the inputs directory, and a tool that merges input directories,
taking the best solutions. We may also provide a C++ skeleton.

If we release these, they will be released by the time that all inputs are
released.


## Ideas from 4/14

This is a NP-hard problem. (Can be reduced to Integer LP) We want the best approximation to optimal solution.

Algorithms we could use:
- k-means clustering: increase k until the max centriod radius is <R_s.
- Use linear programming and convert into Integer solution(what is the approximation bound for this?)
would need to increase number of variables (variables representing towers),running the linear program 
on each number of towers, and find the optimal number of towers
- minimum geometric disk cover: essentially the same problem but without the penalty radius. There is a detailed paper on this and the
approximation bound on it is very good.
-after we have a solution - use a local search heuristic to improve solution (jiggle our solution points)


Minimum geometric disk is our favorite. With local search/jiggling to help with penaulty radius after min geo disk returns its solution.



