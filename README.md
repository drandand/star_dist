# STAR_DIST - Star Distribution

## DESCRIPTION

This project contains python code to define a collection of star clusters, each comprised of one or more individual components.  These components are defined in a YAML structure (for example see [sample.yaml](sample.yaml)).  The python code is intended to support a basic parsing and visualization capability to provide a visual representation of the resulting star cluster definitions.  It does not actually simulate these cluster, but it is intended to provide specification of the initial conditions for a simulation such as may be seen in [csim](https://github.com/cyber-front/csim).

The parser uses a recursive descent approach to decompose the configuration into actionable elements used to generate one or more clusters to support the simulation.

This is primarily intended as a proof of concept and a tool to assist authoring these initial configuration specifications.

## FILES

### [cluster_common.py](cluster_common.py)
Contains some common elements other modules can reference.

### [cluster_components.py](cluster_components.py)
Contains parser targeting structural elements which would comprise a star cluster.

### [cluster_distributions.py](cluster_distributions.py)
Contains parsers and specifications for defining how to generate random scalar and vector values

### [cluster_parser.py](cluster_parser.py)
Contains specifications for some low-level elements which are repeated in several places

### [cluster_simulation.py](cluster_simulation.py)
Contains the high level specification for the overall simulation which can include one or more star clusters

### [cluster.py](cluster.py)
Contains the features defining the composition of several components to generate a single star cluster

### [star_dist.py](star_dist.py)
Main program processing CLI inputs to perform the high level management of the parser.

### [requirements.yaml](requirements.yaml)
Contains conda configured list of dependencies for the star_dist project

### [sample.yaml](sample.yaml)
Contains a sample specification for a simulation.
