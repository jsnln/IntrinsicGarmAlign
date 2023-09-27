import argparse
import yaml

from lib.runner import Runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgfile', type=str)
    args = parser.parse_args()

    with open(args.cfgfile, 'r') as cfgfile:
        cfg = yaml.safe_load(cfgfile)

    runner = Runner(cfg)
    runner.compute_eigenfuncs()
    runner.set_template()
    runner.pose_by_lbs()
    runner.deform_coarse_stage()
    runner.rectify_embeddings_by_FM()
    runner.deform_refining_stage()
    runner.shape_transfer()