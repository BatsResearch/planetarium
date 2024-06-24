# planetarium

Planetarium is a benchmark for assessing LLMs in translating natural language descriptions of planning problems into PDDL.

## Installation
To install the `planetarium` package, you can use the following command:
```bash
pip install git+https://github.com/BatsResearch/planetarium.git
```

For development or using our evaluate & finetune scripts, you can clone the repository and install all dependencies using the following commands:
```bash
git clone https://github.com/BatsResearch/planetarium.git
cd planetarium
poetry install --with all
```

To use `planetarium.downward`, you will need to have the [Fast-Downward](https://www.fast-downward.org/) planner installed, and the [VAL](https://github.com/KCL-Planning/VAL) plan validator. The following commands is one way to install them with minimal overhead:
```bash
# Fast-Downward via Apptainer
apptainer pull fast-downward.sif docker://aibasel/downward:latest
# VAL download link might not work, follow instructions to download binary at: https://github.com/KCL-Planning/VAL
mkdir tmp
curl -o tmp/VAL.zip https://dev.azure.com/schlumberger/4e6bcb11-cd68-40fe-98a2-e3777bfec0a6/_apis/build/builds/77/artifacts?artifactName=linux64\&api-version=7.1\&%24format=zip
unzip tmp/VAL.zip -d tmp/
tar -xzvf tmp/linux64/*.tar.gz -C tmp/ --strip-components=1
# clean up
rm -rf tmp
# Make sure to add fast-downward.sif and VAL to your PATH or make aliases.
```