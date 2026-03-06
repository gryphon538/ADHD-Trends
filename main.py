from typing import Any

import numpy as np

from nilearn import datasets
from nilearn.plotting import show
from nilearn import plotting

from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
)
from nilearn.maskers import NiftiSpheresMasker

# import matplotlib.pyplot as plt

adhd_dataset = datasets.fetch_adhd(
    n_subjects=40, data_dir=None, url=None, resume=True, verbose=1
)


pcc_coords = (0, -53, 26)


def interpret_dataset() -> Any:
    """A method for sorting the data"""

    query = 70

    while query != 1 and query != 0:
        query = int(input("Enter 1 for ADHD results and 0 for neurotypical ones. \n"))

        if query == 1 or query == 0:
            excluded_dataset = adhd_dataset.phenotypic[
                adhd_dataset.phenotypic["adhd"] == query
            ]

        else:
            print("That is not a valid input. Please try again.")

    filtered_dataset = excluded_dataset[
        [
            "Subject",
            "age",
            "sex",
            "full_4_iq",
            "viq",
            "piq",
            "adhd_inattentive",
            "adhd_combined",
            "oppositional",
            "cog_inatt",
            "dsm_iv_tot",
            "dsm_iv_inatt",
            "dsm_iv_h_i",
            "conn_adhd",
            "conn_gi_tot",
            "adhd",
        ]
    ]

    return filtered_dataset


def glm_analysis(x: int) -> None:
    """Analyzes the dataset info"""

    seed_masker = NiftiSpheresMasker(
        [pcc_coords],
        radius=10,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=adhd_dataset.t_r,
        memory="nilearn_cache",
        memory_level=1,
        verbose=1,
    )
    seed_time_series = seed_masker.fit_transform(adhd_dataset.func[x])

    n_scans = seed_time_series.shape[0]
    frametimes = np.linspace(0, (n_scans - 1) * adhd_dataset.t_r, n_scans)

    design_matrix = make_first_level_design_matrix(
        frametimes,
        hrf_model="spm",
        add_regs=seed_time_series,
        add_reg_names=["pcc_seed"],
    )

    dmn_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    contrasts = {"seed_based_glm": dmn_contrast}

    first_level_model = FirstLevelModel(verbose=1)
    first_level_model = first_level_model.fit(
        run_imgs=adhd_dataset.func[x], design_matrices=design_matrix
    )

    z_map = first_level_model.compute_contrast(
        contrasts["seed_based_glm"], output_type="z_score"
    )

    display = plotting.plot_stat_map(
        z_map, threshold=3.0, title=f"GLM for Subject {x}", cut_coords=pcc_coords
    )
    display.add_markers(marker_coords=[pcc_coords], marker_color="g", marker_size=300)

    show()


filtered = interpret_dataset()

for i in filtered.index:
    glm_analysis(i)
