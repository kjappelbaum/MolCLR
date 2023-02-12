import fire 
import subprocess
SUBMISSION_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}      # Job name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=32gb                    # Job memory request
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
#SBATCH --qos=serial

python finetune_lc_cli.py {data} {size}
"""

datasets = ['esol', 'esol_2', 'esol_5']
sizes = [10, 20, 50, 100, 200, 500]


if __name__ == '__main__':
    for dataset in datasets:
        for size in sizes:
            with open(f'submit_{dataset}_{size}.sh', 'w') as f:
                f.write(SUBMISSION_TEMPLATE.format(name=f'{dataset}_{size}', data=dataset, size=size))
            subprocess.run(['sbatch', f'submit_{dataset}_{size}.sh'])
