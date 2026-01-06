

def get_mass_dict(base_mass):
    """ Generate the dictionary with mass distribution based on 
        S. Plagenhoef, F. G. Evans, and T. Abdelnour, “Anatomical Data for Analyzing Human
        Motion,” Research Quarterly for Exercise and Sport, vol. 54, no. 2, pp. 169-178, 1983.

        Args:
            base_mass (float): The mass of the whole person to be segmented into the body distribution
        """
    head = 8.2 * base_mass / 100
    trunk = 53.2 * base_mass / 100
    thorax = 17.02 * base_mass / 100
    abdomen = 12.24 * base_mass / 100
    pelvis = 15.96 * base_mass / 100
    total_arm = 4.97 * base_mass / 100
    upper_arm = 2.9 * base_mass / 100
    forearm = 1.57 * base_mass / 100
    hand = 0.5 * base_mass / 100
    forearm_hand = 2.07 * base_mass / 100
    total_leg = 18.43 * base_mass / 100
    thigh = 11.75 * base_mass / 100
    leg = 5.35 * base_mass / 100
    foot = 1.33 * base_mass / 100
    leg_foot = 6.68 * base_mass / 100

    mass_dict = {
        "head_up": head * 2 / 5,
        "head_down": head * 2 / 5,
        "neck": head / 5,
        "shoulder": thorax * 3 / 14,
        "body2": thorax * 4 / 7,
        "body3": abdomen,
        "body4": pelvis * 2 / 3,
        "body5": pelvis / 3 + thigh / 2,
        "upper_arm": upper_arm,
        "lower_arm": forearm,
        "hand": hand,
        "upper_leg": thigh * 3 / 4,
        "lower_leg": leg,
        "foot": foot,
        "total": base_mass
    }

    return mass_dict