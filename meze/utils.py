def residue_restraint_mask(residue_ids: list[int]) -> str:
    """Generate an Amber-style restraint mask.

    Adapted from: 
    https://github.com/OpenBioSim/biosimspace/blob/devel/python/BioSimSpace/_Config/_amber.py#L203
    
    Args:
        residue_ids (list[int]): list of residue ids that should be restrained

    Returns:
        str: amber-style residue id-based restraint mask
    """
    residue_ids = list(set(residue_ids))
    residue_ids.sort()

    if len(residue_ids) == 1:
        restraint_mask = f"{residue_ids[0]}"
    else:
        restraint_mask = f"{residue_ids[0]}"
        previous_id = residue_ids[0]
        lead_id = previous_id

        for idx in residue_ids[1:]:
            if idx - previous_id > 1:
                if previous_id != lead_id:
                    restraint_mask += f"{previous_id},{idx}"
                else:
                    restraint_mask += f",{idx}"
                lead_id = idx
            else:
                if idx - lead_id == 1:
                    restraint_mask += "-"
            previous_id = idx
        if idx - residue_ids[-2] == 1:
            restraint_mask += f"{idx}"
        else:
            if idx != lead_id:
                restraint_mask += f",{idx}"

    return restraint_mask