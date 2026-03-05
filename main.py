from nilearn import datasets

# import matplotlib.pyplot as plt

adhd_dataset = datasets.fetch_adhd(
    n_subjects=40, data_dir=None, url=None, resume=True, verbose=1
)

pcc_coords = (0, -53, 26)


def interpret_dataset() -> None:
    """stop yelling at me linter"""

    excluded_dataset = adhd_dataset.phenotypic[adhd_dataset.phenotypic["adhd"] == 1]

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
        ]
    ]
    print(filtered_dataset)
    # print(adhd_dataset.phenotypic.to_string())


interpret_dataset()
