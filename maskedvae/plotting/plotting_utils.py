import numpy as np
import matplotlib.pyplot as plt


def cm2inch(cm, cm2=None, INCH=2.54):
    """Convert cm to inch"""
    if isinstance(cm, tuple):
        return tuple(i / INCH for i in cm)
    elif cm2 is not None:
        return cm / INCH, cm2 / INCH
    else:
        return cm / INCH


def make_col_dict(task="fly", n_masks=5, samemask=False):
    if task == "fly":
        col_dict = {
            mask: {masked: "blue" for masked in [True, False]}
            for mask in ["mask_left_claws", "all_obs"]
        }
        col_dict["mask_left_claws"][True] = "darkred"
        col_dict["mask_left_claws"][False] = "lightcoral"
        col_dict["all_obs"][True] = "midnightblue"
        col_dict["all_obs"][False] = "dodgerblue"
    elif task == "gaussian":
        print(f"mask {n_masks-2} is all observed data, mask  {n_masks-1} is new data")
        # return 3 dark red tones
        red_col = [
            "darkred" if samemask else "darksalmon",
            "darkred",
            "darkred" if samemask else "tomato",
            "maroon",
            "indianred",
            "salmon",
            "firebrick",
            "mistyrose",
        ]
        blue_col = [
            "midnightblue" if samemask else "blue",
            "midnightblue",
            "midnightblue" if samemask else "slateblue",
            "steelblue",
            "cornflowerblue",
            "lightsteelblue",
            "aliceblue",
            "darkblue",
        ]

        if len(red_col) < (n_masks - 2):
            while len(red_col) < n_masks:
                red_col = red_col + red_col
                blue_col = blue_col + blue_col

        # all obs and new data
        last_colors_red = ["lightcoral", "black"]
        last_colors_blue = ["dodgerblue", "grey"]

        # ensure the last two colors for all obs and new mask are the same indeoendent of number of masks
        red_col = red_col[: n_masks - 2] + last_colors_red
        blue_col = blue_col[: n_masks - 2] + last_colors_blue

        col_dict = {
            mask: {masked: "blue" for masked in range(n_masks)}
            for mask in [
                "zero_imputation",
                "zero_imputation_mask_concatenated_encoder_only",
            ]
        }
        for mask_n in range(n_masks):
            col_dict["zero_imputation"][mask_n] = blue_col[mask_n]
            col_dict["zero_imputation_mask_concatenated_encoder_only"][
                mask_n
            ] = red_col[mask_n]

    return col_dict


def make_col_dict_new(task="fly", n_masks=5, samemask=False):
    if task == "fly":
        col_dict = {
            mask: {masked: "blue" for masked in [True, False]}
            for mask in ["mask_left_claws", "all_obs"]
        }
        col_dict["mask_left_claws"][True] = "darkred"
        col_dict["mask_left_claws"][False] = "lightcoral"
        col_dict["all_obs"][True] = "midnightblue"
        col_dict["all_obs"][False] = "dodgerblue"
    elif task == "gaussian":
        print(f"mask {n_masks-2} is all observed data, mask  {n_masks-1} is new data")
        # return 3 dark red tones
        red_col = [
            "darkred" if samemask else "#991E29",
            "darkred",
            "darkred" if samemask else "#601A26",
            "maroon",
            "indianred",
            "salmon",
            "firebrick",
            "mistyrose",
        ]
        blue_col = [
            "midnightblue" if samemask else "#234558",
            "midnightblue",
            "midnightblue" if samemask else "#1E3B48",
            "#383B57",
            "cornflowerblue",
            "lightsteelblue",
            "aliceblue",
            "darkblue",
        ]

        if len(red_col) < (n_masks - 2):
            while len(red_col) < n_masks:
                red_col = red_col + red_col
                blue_col = blue_col + blue_col

        # all obs and new data
        last_colors_red = ["lightcoral", "black"]
        last_colors_blue = ["dodgerblue", "grey"]

        # ensure the last two colors for all obs and new mask are the same indeoendent of number of masks
        red_col = red_col[: n_masks - 2] + last_colors_red
        blue_col = blue_col[: n_masks - 2] + last_colors_blue

        col_dict = {
            mask: {masked: "blue" for masked in range(n_masks)}
            for mask in [
                "zero_imputation",
                "zero_imputation_mask_concatenated_encoder_only",
            ]
        }
        for mask_n in range(n_masks):
            col_dict["zero_imputation"][mask_n] = blue_col[mask_n]
            col_dict["zero_imputation_mask_concatenated_encoder_only"][
                mask_n
            ] = red_col[mask_n]

    return col_dict


def make_label_dict(task="fly", n_masks=5, mask_nr=False):

    if task == "fly":
        label_dict = {
            mask: {masked: "blue" for masked in [True, False]}
            for mask in ["mask_left_claws", "all_obs"]
        }
        label_dict["mask_left_claws"][True] = "masked mask"
        label_dict["mask_left_claws"][False] = "masked obs"
        label_dict["all_obs"][True] = "naive mask"
        label_dict["all_obs"][False] = "naive obs"

    elif task == "gaussian":
        label_dict = {
            mask: {masked: "blue" for masked in range(n_masks)}
            for mask in [
                "zero_imputation",
                "zero_imputation_mask_concatenated_encoder_only",
            ]
        }

        for mask_n in range(n_masks):
            label_dict["zero_imputation"][mask_n] = (
                f"naive mask {mask_n}" if mask_nr else f"naive mask"
            )
            label_dict["zero_imputation_mask_concatenated_encoder_only"][mask_n] = (
                f"masked mask {mask_n}" if mask_nr else f"masked mask"
            )

        label_dict["zero_imputation_mask_concatenated_encoder_only"][
            n_masks - 2
        ] = "masked obs"
        label_dict["zero_imputation"][n_masks - 2] = "naive obs"

        label_dict["zero_imputation_mask_concatenated_encoder_only"][
            n_masks - 1
        ] = "masked new mask"
        label_dict["zero_imputation"][n_masks - 1] = "naive new mask"
    return label_dict


def create_index_mapping(permuted):
    """ensure the order of the data conditions is the same for different ordering of runs"""
    # condition indices order used in the manuscript
    original = [1.0732682811326433, 1.076781774405591, 1.233158650412952]

    #  map values from the original list to their indices in the permuted list
    value_to_permuted_index = {value: idx for idx, value in enumerate(permuted)}

    # make mapping
    index_mapping = {
        idx: value_to_permuted_index[value] for idx, value in enumerate(original)
    }

    return index_mapping
