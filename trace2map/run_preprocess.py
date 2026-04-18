
import os
import shutil
import random
import glob
import torch
import numpy as np
import json
from tqdm import tqdm
from utils.data_utils import preprocess_dataset, rasterize_csv_files

# ==========================================
# ==========================================

DATA_SOURCE = 'sumo'

COLLECTOR_OUTPUT_DIR = r'../dataset_collector/output'
SUMO_DATA_DIR = 'data/sumo_data'

RAW_DATA_DIR = r'carla_data_collection\Workzone Drivable Area Mapping Dataset'
PROCESSED_DIR = 'data/processed_data'

METADATA_PATH = 'data/scenario_metadata.csv'

SCENE_SAMPLING = {
    5: 10,
    10: 10,
    15: 10,
    20: 10,
    25: 10,
    30: 10,
}
OBS_THRESHOLD = 0.1

MODE = "all"

LEAVE_OUT_SCENARIO = 'Town03_2L_Close_1'
SINGLE_SCENARIO = 'Town04_4L_Close_1'

LONGITUDINAL_PADDING = 30.0
LATERAL_PADDING = 5.0


def main():
    if DATA_SOURCE == 'sumo':
        from utils.sumo_data_adapter import convert_sumo_to_cvae_format
        raw_data_dir, metadata_path = convert_sumo_to_cvae_format(
            COLLECTOR_OUTPUT_DIR, SUMO_DATA_DIR
        )
        processed_dir = SUMO_DATA_DIR.rstrip('/').rstrip('\\')
    elif DATA_SOURCE == 'human':
        raw_data_dir = RAW_DATA_DIR
        metadata_path = METADATA_PATH
        processed_dir = PROCESSED_DIR
    else:
        print(f"\nError: Unknown DATA_SOURCE '{DATA_SOURCE}'")
        print("      Options: 'human', 'sumo'")
        return

    print("\n" + "=" * 70)
    print(f"Multi-scenario preprocessing script")
    print(f"Data source:     {DATA_SOURCE}")
    print(f"Experiment mode: {MODE}")
    print(f"Longitudinal padding:   {LONGITUDINAL_PADDING} m")
    print(f"Lateral padding:   {LATERAL_PADDING} m")
    print("=" * 70)

    if not os.path.exists(raw_data_dir):
        print(f"\nError: Cannot find dataset directory '{raw_data_dir}'")
        print("Please verify the path is correct (should point to the root directory containing scenario subdirectories)")
        return

    print(f"\n[Scan] Scanning dataset directory...")
    all_scenarios = []
    for item in os.listdir(raw_data_dir):
        item_path = os.path.join(raw_data_dir, item)
        if os.path.isdir(item_path):
            csv_count = sum(1 for f in os.listdir(item_path) if f.endswith('.csv'))
            if csv_count > 0:
                all_scenarios.append(item)

    all_scenarios.sort()
    print(f"[Scan] Found {len(all_scenarios)} scenarios:")
    for i, sc in enumerate(all_scenarios, 1):
        print(f"        {i:2d}. {sc}")

    if len(all_scenarios) == 0:
        print(f"\nError: No subdirectories containing CSV files found in '{raw_data_dir}'!")
        return

    if MODE == "all":
        scenarios_to_process = None
        print(f"\n[Mode] Will process all {len(all_scenarios)} scenarios")

    elif MODE == "leave_one_out":
        if LEAVE_OUT_SCENARIO not in all_scenarios:
            print(f"\nWarning: Specified leave-out scenario '{LEAVE_OUT_SCENARIO}' not found in dataset")
            print(f"      Will randomly select a scenario to exclude")
            LEAVE_OUT_SCENARIO = random.choice(all_scenarios)

        scenarios_to_process = [s for s in all_scenarios if s != LEAVE_OUT_SCENARIO]
        print(f"\n[Mode] Leave-One-Out: Excluding '{LEAVE_OUT_SCENARIO}'")
        print(f"      Will process remaining {len(scenarios_to_process)} scenarios")
        print(f"      Suggest setting TARGET_SCENARIO = '{LEAVE_OUT_SCENARIO}' in adapt.py")

    elif MODE == "single_adapt":
        if SINGLE_SCENARIO not in all_scenarios:
            print(f"\nWarning: Specified scenario '{SINGLE_SCENARIO}' not found in dataset")
            print(f"      Will use the first scenario")
            SINGLE_SCENARIO = all_scenarios[0]

        scenarios_to_process = [SINGLE_SCENARIO]
        print(f"\n[Mode] Single Scenario: Only processing '{SINGLE_SCENARIO}'")
        other_scenarios = [s for s in all_scenarios if s != SINGLE_SCENARIO]
        if other_scenarios:
            adapt_suggestion = random.choice(other_scenarios)
            print(f"      Suggest setting TARGET_SCENARIO = '{adapt_suggestion}' in adapt.py")

    else:
        print(f"\nError: Unknown mode '{MODE}'")
        print("      Options: 'all', 'leave_one_out', 'single_adapt'")
        return

    if DATA_SOURCE == 'sumo':
        if os.path.exists(processed_dir):
            removed = 0
            for f in os.listdir(processed_dir):
                if f.endswith('.pt') or f == 'global_bounds.json':
                    p = os.path.join(processed_dir, f)
                    if os.path.isfile(p):
                        os.remove(p)
                        removed += 1
            if removed > 0:
                print(f"\n[Cleanup] Removed old preprocessed files: {removed} (.pt / global_bounds.json)")
        os.makedirs(processed_dir, exist_ok=True)
    else:
        if os.path.exists(processed_dir):
            print(f"\n[Cleanup] Removing old output directory '{processed_dir}'...")
            shutil.rmtree(processed_dir)
        os.makedirs(processed_dir)

    print(f"\n[Preprocess] Starting processing...")
    try:
        count = preprocess_dataset(
            raw_data_dir=raw_data_dir,
            output_dir=processed_dir,
            scenarios_to_process=scenarios_to_process,
            metadata_path=metadata_path,
            longitudinal_padding=LONGITUDINAL_PADDING,
            lateral_padding=LATERAL_PADDING
        )

        if count > 0:
            print(f"\n{'='*70}")
            print(f"Success! Generated {count} .pt files")
            print(f"  Output directory: {processed_dir}")
            print(f"\nNext steps:")
            print(f"  1. Run 'python train.py' to train the model")
            print(f"  2. After training, edit 'adapt.py' to set TARGET_SCENARIO")
            print(f"  3. Run 'python adapt.py' to perform few-shot adaptation")
            print(f"{'='*70}\n")
        else:
            print(f"\nFailed: Could not generate any .pt files")
            print(f"Please check the dataset path and CSV format")

    except Exception as e:
        print(f"\nError: Preprocessing failed")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def generate_sampled_scenes(processed_dir, raw_data_dir, scene_sampling, obs_threshold):
    """
    Generate sampling samples for each scenario.

    Args:
        processed_dir: Processed data directory (containing *_GT.pt files)
        raw_data_dir: Raw data directory (containing scenario CSV files)
        scene_sampling: dict {k: count} sampling configuration
        obs_threshold: float obs_mask generation threshold

    Returns:
        total_samples: int total number of generated samples
    """
    print("\n" + "=" * 70)
    print("Starting scene sampling sample generation")
    print("=" * 70)

    bounds_path = os.path.join(processed_dir, 'global_bounds.json')
    if not os.path.exists(bounds_path):
        print(f"Error: Cannot find global bounds file {bounds_path}")
        return 0

    with open(bounds_path, 'r') as f:
        global_bounds = json.load(f)
    print(f"[1/4] Loaded global bounds: xmin={global_bounds['xmin']:.2f}, xmax={global_bounds['xmax']:.2f}")

    gt_files = glob.glob(os.path.join(processed_dir, '*_GT.pt'))
    print(f"[2/4] Found {len(gt_files)} GT files")

    if len(gt_files) == 0:
        print("Error: No GT files found, please run preprocessing first")
        return 0

    total_samples = 0

    print(f"[3/4] Starting sampling sample generation...")
    for gt_file in tqdm(gt_files, desc="      Processing scenarios"):
        gt_data = torch.load(gt_file)
        scenario_name = gt_data['scenario_name']
        gt_heatmap = gt_data['heatmap']  # [1, 512, 128]
        num_lanes = gt_data['num_lanes']
        alignment_info = gt_data['alignment_info']

        scene_dir = os.path.join(raw_data_dir, scenario_name)
        if not os.path.exists(scene_dir):
            print(f"\n      Warning: Cannot find scenario directory {scene_dir}, skipping")
            continue

        csv_files = glob.glob(os.path.join(scene_dir, '*.csv'))
        if len(csv_files) == 0:
            print(f"\n      Warning: Scenario {scenario_name} has no CSV files, skipping")
            continue

        for k, count in scene_sampling.items():
            if k > len(csv_files):
                print(f"\n      Warning: Scenario {scenario_name} only has {len(csv_files)} CSVs, cannot sample k={k}, skipping")
                continue

            for sample_idx in range(count):
                sampled_csvs = random.sample(csv_files, k)

                obs_heatmap_np = rasterize_csv_files(sampled_csvs, alignment_info, global_bounds)
                obs_heatmap = torch.from_numpy(obs_heatmap_np).unsqueeze(0).float()  # [1, 512, 128]

                obs_mask = (obs_heatmap > obs_threshold).float()

                sample_data = {
                    'obs_heatmap': obs_heatmap,
                    'obs_mask': obs_mask,
                    'gt_heatmap': gt_heatmap,
                    'num_lanes': num_lanes,
                    'scenario_name': scenario_name,
                    'k': k,
                    'traj_files': [os.path.basename(f) for f in sampled_csvs]
                }

                safe_name = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in scenario_name])
                sample_filename = f"{safe_name}_k{k}_s{sample_idx+1:03d}.pt"
                torch.save(sample_data, os.path.join(processed_dir, sample_filename))
                total_samples += 1

    print(f"\n[4/4] Done! Generated {total_samples} sampling samples in total")
    print("=" * 70)
    return total_samples


if __name__ == '__main__':
    main()

    if DATA_SOURCE == 'sumo':
        _processed_dir = SUMO_DATA_DIR.rstrip('/').rstrip('\\')
        _raw_data_dir = SUMO_DATA_DIR
    else:
        _processed_dir = PROCESSED_DIR
        _raw_data_dir = RAW_DATA_DIR

    print("\n" + "=" * 70)
    print("Phase 2: Generating scene sampling samples")
    print("=" * 70)

    try:
        sample_count = generate_sampled_scenes(
            processed_dir=_processed_dir,
            raw_data_dir=_raw_data_dir,
            scene_sampling=SCENE_SAMPLING,
            obs_threshold=OBS_THRESHOLD
        )

        if sample_count > 0:
            print(f"\nSuccessfully generated {sample_count} sampling samples")
            print(f"  Expected: {len(SCENE_SAMPLING)} k values x samples per scenario")
            print(f"  Config: {SCENE_SAMPLING}")
        else:
            print("\nFailed: Could not generate sampling samples")

    except Exception as e:
        print(f"\nError: Sampling sample generation failed")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
