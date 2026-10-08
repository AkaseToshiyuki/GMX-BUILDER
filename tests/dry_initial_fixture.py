"""Synthetic admission records, not evidence of physical validation."""


def dry_initial_fixture():
    return {
        "policy": "complete-lipid-z-envelope-v1",
        "margin_nm": 0.0011,
        "files": {
            name: {
                "sha256": digest * 64,
                "lipid_atoms": 128,
                "water_sites": 300,
                "water_sites_in_membrane": 0,
                "lipid_z_bounds_nm": [2.0, 6.0],
            }
            for name, digest in (("solvated.gro", "a"), ("ionized.gro", "b"))
        },
    }
