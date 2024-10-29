from typing import Literal
import numpy as np
import re
import sys
from tabulate import tabulate

np.set_printoptions(precision=1)
DATASET: Literal["tartanair", "eth3d"] = "tartanair"

if __name__ == '__main__':
    if DATASET == "tartanair":
        categories = {
            "Indoors": ["carwelding_Easy", "hospital", "japanesealley", "office", "office2"],
            "Outdoors": ["abandonedfactory", "abandonedfactory_night", "amusement", "endofworld_Hard", "gascola", "ocean", "neighborhood", "oldtown", "seasonsforest_winter", "seasidetown", "soulcity", "westerndesert_Hard"],

            "Natural": ["gascola", "ocean", "seasonsforest_winter"],
            "Artificial": ["abandonedfactory", "abandonedfactory_night", "carwelding_Easy", "endofworld_Hard", "hospital", "japanesealley", "office", "office2", "soulcity"],
            "Mixed": ["amusement", "neighborhood", "oldtown", "seasidetown", "westerndesert_Hard"],
        }
        categories["Easy"] = [(v if 'Easy' in v else v+'_Easy') for v in categories['Indoors']+ categories['Outdoors']]
        categories["Hard"] = [(v if 'Hard' in v else v+'_Hard') for v in categories['Indoors']+ categories['Outdoors']]
        categories["Overall"] = categories["Indoors"] + categories["Outdoors"]
        starting_line_regex = re.compile(".+STARTING (\w+) .+ (Easy|Hard).+")
    elif DATASET == "eth3d":
        categories = {
            "cables": "[\'cables_2+cables_3+cables_4+cables_5\',\'cables_1\',\'cables_1\']",
            "camera_shake": "[\'camera_shake_2+camera_shake_3\',\'camera_shake_1\',\'camera_shake_1\']",
            "ceiling": "[\'ceiling_2\',\'ceiling_1\',\'ceiling_1\']",
            "desk": "[\'desk_2+desk_1\',\'desk_3\',\'desk_3\']",
            "desk_changing": "[\'desk_changing_2\',\'desk_changing_1\',\'desk_changing_1\']",
            "einstein": "[\'einstein_2\',\'einstein_1\',\'einstein_1\']",
            "einstein_global_light_changes": "[\'einstein_global_light_changes_2+einstein_global_light_changes_3\',\'einstein_global_light_changes_1\',\'einstein_global_light_changes_1\']",
            "mannequin": "[\'mannequin_1+mannequin_3+mannequin_7\',\'mannequin_4+mannequin_5\',\'mannequin_4+mannequin_5\']",
            "mannequin_face": "[\'mannequin_face_2+mannequin_face_3\',\'mannequin_face_1\',\'mannequin_face_1\']",
            "planar": "[\'planar_2+planar_3\',\'planar_1\',\'planar_1\']",
            "plant": "[\'plant_2+plant_3+plant_5\',\'plant_1+plant_4\',\'plant_1+plant_4\']",
            "plant_scene": "[\'plant_scene_2+plant_scene_3\',\'plant_scene_1\',\'plant_scene_1\']",
            "sfm_lab_room": "[\'sfm_lab_room_1\',\'sfm_lab_room_2\',\'sfm_lab_room_2\']",
            "sofa": "[\'sofa_1+sofa_4\',\'sofa_2+sofa_3\',\'sofa_2+sofa_3\']",
            "table": "[\'table_1+table_2+table_5+table_6\',\'table_3+table_4+table_7\',\'table_3+table_4+table_7\']",
            "vicon_light": "[\'vicon_light_2\',\'vicon_light_1\',\'vicon_light_1\']",
            "large_loop": "[\'large_loop_2+large_loop_3\',\'large_loop_1\',\'large_loop_1\']",
        }
        categories = {k: [v] for k, v in categories.items()}
        categories["Overall"] = [v[0] for k, v in categories.items()]

    eval_results = {}
    for name in sorted(sys.argv[1:]):
        with open(name) as f:
            lines = f.readlines()
            cur_result_key = None
            for idx, line in enumerate(lines):
                if "STARTING" in line:
                    if '[' in line and ']' in line:
                        for cat, keywords in categories.items():
                            for kw in keywords:
                                if kw in line:
                                    cur_result_key = kw
                                    break
                        assert cur_result_key is not None, line
                        results = eval_results.setdefault(cur_result_key, {})
                        n_seq = -1
                    else:
                        env, diff = starting_line_regex.match(line).groups()
                        cur_result_key = f"{env}_{diff}"
                        results = eval_results.setdefault(cur_result_key, {})
                        n_seq = -1
                if "reproj_mma/" in line or 'pose_auc/' in line:
                    # split by the `|` and white space characters
                    tmp = ''.join(line.split("│")).split()
                    metric_name, value = tmp[0], float(tmp[1])
                    results.setdefault(metric_name, [])
                    if value not in results[metric_name]:
                        results[metric_name].append(value)
                if "DataLoader " in line and "Test metric" in line:
                    # n_seq = max(int(loader_id) for loader_id in re.findall(".+DataLoader ([0-9]+)", line)) + 1
                    if "n_seq" not in results:
                        results["n_seq"] = 0
                    results["n_seq"] += 1
    

    mma = {cat: {} for cat in categories}
    num = {cat: 0 for cat in categories}
    for series, results in eval_results.items():
        for cat, keywords in categories.items():
            if any(kw in series.rsplit("_", 1) for kw in keywords) or any(series == kw for kw in keywords):
                if cat == 'Mixed':
                    print(series)
                for metric_name in results:
                    if metric_name == "n_seq":
                        num[cat] += 1
                    else:
                        mma[cat].setdefault(metric_name, np.zeros(len(results[metric_name])))
                        mma[cat][metric_name] += np.array(results[metric_name]).mean() *100

    table_rows = []
    for cat, cat_result in mma.items():
        table_rows.append([cat] + [num[cat]] + [metric_data / num[cat] for metric_data in cat_result.values()])
    print(tabulate(table_rows, headers=["Split"] + list(eval_results[next(iter(eval_results.keys()))].keys()), floatfmt=".1f"))
