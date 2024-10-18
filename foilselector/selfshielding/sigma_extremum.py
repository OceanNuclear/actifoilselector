# class used to find the max microscopic cross-section value

__all__ = ["MaxSigma", "sigma_to_thickness"]
class MaxSigma(dict):
    def __getitem__(self, parent_product_mt):
        """
        key is provided in the format of Pt206-Pt207-MT=(102,5).
        we will then return  up the following:
        
        max([max_sigma["Pt206-Pt207-MT=102"], max_sigma["Pt206-Pt207-MT=5"]])
        """
        parent_product_, mts = parent_product_mt.split("=")
        results = []
        for mt in mts.strip("()").split(","):
            results.append(super(MaxSigma, self).__getitem__(parent_product_ +"="+ mt))
        return max(results)

def sigma_to_thickness(sigma, num_density):
    """Calculates the thickness where the probability of reaction of atoms reaches 1.0.
    
    Probability of absorption per cm = (sigma*1E24) * (number density in cm^-3)
    consider three BCC materials, all with the same microscopic reaction cross-section.
    mat1 = 1 barn, density(homogeneous) = 1E21cm^-3;    (a=1E-7, b=1E-7, c=1E-7) ∴ P(rx per cm) = 0.001
    mat2 = 1 barn, density(non-homogeneous) = 1E22cm^-3;(a=1E-8, b=1E-7, c=1E-7) ∴ P(rx per cm) = 0.01
    mat3 = 1 barn, density(homogeneous) = 1E24cm^-3;    (a=1E-8, b=1E-8, c=1E-8) ∴ P(rx per cm) = 1
                                                        (where a,b,c are lattice spacing in cm)
    P1 = P(rx per cm of mat1)
    P2 = P(rx per cm of mat2) = 10*P1
    P3 = P(rx per cm of mat3) = 1000*P1

    Number of atoms that a neutron passes through before having expected :
        mat1: 1000 cm = 1000/1E-7 = 1E10
        mat2: 100 cm = 100/1E-8 = 1E10
        mat3: 1 cm = 1/1E-8 = 1E8
    This shows that the quantity in the section immediately above ^ is dependent on number density.

    Therefore this function can only be used in a scenario where MaxSigma and number density of atoms are both known.
    In such a situation, we would care about the thickness in cm more than thickness in number of atoms.
    Therefore, I've opted to let this function return max.thickness in cm.
    Returns
    -------
    thickness in cm such that P(reaction) = 1 after a neutron passes through the foil perpendicularly.
    """
    macroscopic_xs = sigma * num_density
    return 1/macroscopic_xs # thickness 
