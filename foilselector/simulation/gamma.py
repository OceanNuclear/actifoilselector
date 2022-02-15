"""
DecayPhoton -> (XRayLine, GammaLine, MergedLine)
GammaSpectrumABC -> SingleDecayGammaSignature
    -(multiple decays of the same isotope)->GammaSignature
    -(multiple isotopes, multiple pathways per isotope)->GammaSpectrum
    -(multiple isotopes, ignoring pathways)->MergedGammaSpectrum
"""
from collections import namedtuple
import itertools

from numpy import array as ary
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import uncertainties as unc

from foilselector.generic import ordered_set
ln2 = np.log(2)
arrow = "->"

class DecayPhoton(object):
    def __init__(self, line_dict : dict, parent : str):
        self.__dict__.update(line_dict)
        self.parent = parent

    def __str__(self):
        return "< {} of {} eV from decay of {}>".format(self.__class__.__name__, self.energy, self.parent)

    def __mul__(self, scalar_or_efficiency_curve):
        new_line = self.__class__(self.__dict__.copy(), self.parent)
        if callable(scalar_or_efficiency_curve): # is an efficiency curve
            new_line.intensity = self.intensity * scalar_or_efficiency_curve(unc.nominal_value(self.energy))
        else: # is a scalar
            new_line.intensity = self.intensity * scalar_or_efficiency_curve
        return new_line

    def __add__(self, duplicate_line):
        new_dict = self.__dict__.copy()
        assert self.energy == duplicate_line.energy, "Only allowed to add lines of exactly the same energy"
        new_dict["intensity"] = self.intensity + duplicate_line.intensity
        return self.__class__(new_dict, self.parent)

class XRayLine(DecayPhoton):
    pass

class GammaLine(DecayPhoton):
    pass

class GammaSpectrumABC(object):
    def _init_plot_values(self, default_width=5, unit="keV", force_positive=True):
        """
        Create a triangular gamma peak with width = uncertainty (if there is any) or default width
        unit: unit of the x axis when creating the plot.
        """
        xy = []
        for line in self.lines:
            if isinstance(line.energy, unc.core.AffineScalarFunc):
                xy.append( (line.energy.n - line.energy.s/2, 0) )
                xy.append( (line.energy.n, unc.nominal_value(line.intensity)))
                xy.append( (line.energy.n + line.energy.s/2, 0) )
            else:
                xy.append( (line.energy - default_width/2, 0) )
                xy.append( (line.energy, unc.nominal_value(line.intensity)))
                xy.append( (line.energy + default_width/2, 0) )
        xy = ary(xy).reshape([-1, 2])
        x, y = xy[:, 0], xy[:, 1]
        if force_positive:
            y = np.clip(y, 0, None)
            
        if unit=="eV":
            return x, y
        elif unit=="keV":
            return x/1E3, y
        elif unit=="MeV":
            return x/1E6, y
        else:
            raise TypeError(f"Unit {unit} not accepted!")

    def plot(self, ax=None, default_width=5, sqrt_scale=False, **plot_kwargs):
        """
        Note: This always plots in keV. (This is the only place in the entire program where keV is used instead of eV.)
        This is important to bear in mind when the user use ax.set_xlim to manually set the visualized gamma spectrum window.
        """
        if ax is None:
            ax = plt.subplot()
        x, y = self._init_plot_values(default_width)
        if sqrt_scale:
            line, = ax.plot(x, sqrt(y), **plot_kwargs)
            ax.set_ylabel("sqrt(Intensity)")
        else:
            line, = ax.semilogy(x, y, **plot_kwargs)
            ax.set_ylabel("Intensity")
        ax.set_xlabel(r"$E_\gamma$ (keV)")
        return ax, line

    def list_peaks(self, sort_by=None, isotope_placeholder=""):
        """Sort either by 'intensity'/'counts',
            'energy',
            or None (which would then return the lines in the order that it was read in)"""
        output_list = [LineTuple(line.energy, line.intensity, isotope_placeholder, line) for line in self.lines]
        if sort_by in ("intensity", "counts"):
            return sorted(output_list, key=lambda i:i.intensity, reverse=True)
        elif sort_by=='energy':
            return sorted(output_list, key=lambda i:i.energy)
        else:
            assert sort_by is None or sort_by=="", "Can only accept 'intensity'/'counts', 'energy', and None as arguments to 'sort_by'."
            return output_list

class SingleDecayGammaSignature(GammaSpectrumABC):
    """
    Represents the direct-decay spectrum of ONE atom of a given species.

    Expected to have no repeated peaks, and the peaks are expected to be in ascending order. (though violating these two expectations won't cause an error.)
    """
    def __init__(self, spectrum, isotope):
        """
        Parameters
        ----------
        spectrum: dict accessed by e.g. 'openmc.data.Decay.from_endf("decay/decay/O18").spectra'
        isotope: name of the isotope
        """
        self.isotope = isotope
        self.lines = []
        for rad_type, rad_dict in spectrum.items():
            if rad_type=="xray":
                self.lines.extend(
                [XRayLine(rad, isotope) * rad_dict["discrete_normalization"] for rad in rad_dict["discrete"]]
                )
            elif rad_type=="gamma":
                self.lines.extend(
                [GammaLine(rad, isotope) * rad_dict["discrete_normalization"] for rad in rad_dict["discrete"]]
                )
            # ignore all other types of radiations, including beta, alpha, etc.

    def __mul__(self, multiplier):
        new_lines = [l*multiplier for l in self.lines]
        return GammaSignature(new_lines, self.isotope)

    def __str__(self):
        return "< {} of {} >".format(self.__class__.__name__, self.isotope)

    def __add__(self, another_signature):
        """
        another_signature is assumed to be the signature of the SAME isotope with a differently scaled intensity.
        """
        assert another_signature.isotope==self.isotope, "Can only add together isotopes of the same name."
        # assert ary([unc.nominal_value(line) for line in another_signature.lines])==ary([unc.nominal_value(line) for line in self.lines]), "These two signatures must have exactly the same energies."
        # the energy check is already performed when adding line1+line2
        new_lines = [line1+line2 for line1, line2 in zip(self.lines, another_signature.lines)]
        return GammaSignature(new_lines, self.isotope)

    @classmethod
    def from_openmc_decay(cls, endf_decay_entry):
        """
        create it from a openmc.data.Decay object.
        """
        return cls(endf_decay_entry.spectra, endf_decay_entry.nuclide['name'])

    @classmethod
    def from_endf_file(cls, endf_decay_file_name):
        """
        load up a openmc.data.Decay object from a file name,
        and then access its spectrum using from_openmc_decay
        """
        import openmc.data
        openmc_decay_obj = openmc.data.Decay.from_endf(endf_decay_file_name)
        return cls.from_openmc_decay(openmc_decay_obj)

class GammaSignature(SingleDecayGammaSignature):
    """Created from line_collection"""
    def __init__(self, line_collection, isotope):
        self.isotope = isotope
        self.lines = []
        for line in line_collection: # typechecking.
            if isinstance(line, DecayPhoton):
                self.lines.append(line)
            else:
                raise TypeError("Must provide a list of DecayPhotons (Gamma or Xray) with properly scaled intensities as the argument to line_collections.")

    def plot(self, *args, **kwargs):
        """
        Note: This always plots in keV. (This is the only place in the entire program where keV is used instead of eV.)
        This is important to bear in mind when the user use ax.set_xlim to manually set the visualized gamma spectrum window.
        """
        ax, line = super(GammaSignature, self).plot(*args, **kwargs)
        ax.set_ylabel(ax.get_ylabel().replace("Intensity", "Counts"))
        return ax, line

line_tuple_attributes = ["energy", "intensity", "origin", "photon_type"]
class LineTuple(namedtuple("SortingLine", line_tuple_attributes)):
    def __str__(self):
        str_attr = [str(getattr(self,attr)).rjust(28)[:28] for attr in line_tuple_attributes[:2]]
        str_attr.append(str(self.origin).rjust(9))
        str_attr.append(" "+str(self.photon_type))
        return "("+",".join(str_attr)+")"

class GammaSpectrum(GammaSignature):
    """
    Plots multiple Gamma signature together
    """
    def __init__(self, *signature_collection):
        self.signatures = []
        for sig in signature_collection:
            if type(sig)==GammaSignature:
                self.signatures.append(sig)
            else:
                raise TypeError("Only accept GammaSignature objects")

    def __mul__(self, multiplier):
        new_signatures = [sig*multiplier for sig in self.signatures]
        return GammaSpectrum(*new_signatures)

    def __add__(self, new_signature):
        old_signatures = [sig for sig in self.signatures]
        if not isinstance(new_signature, list):
            new_signature = [new_signature,] # turn into list if not already a list
        for new_sig in new_signature:
            if new_sig.isotope in [old_sig.isotope for old_sig in old_signatures]:
                # get the matching old signature
                old_sig_index = [old_sig_index for old_sig_index, old_sig in enumerate(old_signatures) if old_sig.isotope==new_sig.isotope][0]
                # add the new signature in 
                old_signatures[old_sig_index] += new_sig
            else:
                old_signatures.append(new_sig)
        return GammaSpectrum(*old_signatures)

    def plot(self, ax=None, default_width=5, sqrt_scale=False):
        """
        Note: This always plots in keV. (This is the only place in the entire program where keV is used instead of eV.)
        This is important to bear in mind when the user use ax.set_xlim to manually set the visualized gamma spectrum window.
        """        
        if ax is None:
            ax = plt.subplot()
        line_handles = []
        for sig in self.signatures:
            line_handles.append(sig.plot(ax, default_width, sqrt_scale, label=str(sig.isotope)) [1] )
            # sig.plot() returns (ax, line); so sig.plot()[1] gives the line.
        return ax, line_handles

    def __str__(self):
        return "< {} consists of {} signatures >".format(self.__class__.__name__, len(self.signatures))

    def list_peaks(self, sort_by_intensity=False):
        """
        show the list of peaks as tuples, each tuple containing information about one peak.
        By default this list is sorted by the order .
        Parameters
        ----------
        sort_by_intensity : return the list as descendingly sorted according to intensity
        """
        output_list = []
        for sig in self.signatures:
            output_list.extend(sig.list_peaks(isotope_placeholder=sig.isotope))

        if sort_by in ("intensity", "counts"):
            return sorted(output_list, key=lambda i: i.intensity, reverse=True)
        elif sort_by=="energy":
            return sorted(output_list) # sort according to energy
        else:
            assert sort_by is None or sort_by=="", "Can only accept 'intensity'/'counts', 'energy', and None as arguments to 'sort_by'."
            return output_list

    def sort_signatures_by_intensity(self):
        """
        Rearranges the signatures.
        Warning: if multiple signatures have identical isotope names, only one will be kept.
        """
        origins = [line.origin for line in self.list_peaks(sort_by=None)]
            
        new_signatures = []
        while len(self.signatures)>0 and len(origins)>0:
            # find the matching index
            target_iso = origins[0]
            indices = [ind for ind, sig in enumerate(self.signatures) if sig.isotope==target_iso]

            if len(indices)==1:
                new_signatures.append(self.signatures.pop(indices[0]))
            # and then pop it out of the list.
            origins.pop(0) # remove
        self.signatures = new_signatures
        return

    def merge_into_amophorus_spectrum(self):
        energy, intensity, source_isotope, _photon_type = ary(self.list_peaks(sort_by_intensity=False)).T.reshape([4, -1])
        energy = ary([unc.nominal_value(e) for e in energy])
        if isinstance(source_isotope[0], list):
            source_isotope = ary([arrow.join(isotopes) for isotopes in source_isotope])

        new_lines = []
        for E in sorted(set(energy)):
            mask = energy==E
            sum_intensity = intensity[mask].sum()
            new_photon_dict = {"energy":E,
                                "intensity": sum_intensity,
                                "origin": { ori: intense for ori, intense, in zip(source_isotope[mask], intensity[mask]) }}
            new_lines.append(DecayPhoton(new_photon_dict))
        return MergedGammaSpectrum(new_lines, ordered_set(source_isotope)) # ordered_set is a list

class MergedGammaSpectrum(GammaSpectrum):
    """
    the line inside it's self.lines should each have an attribute of "origin" which is a dict.
    This class is used ONLY for plotting (in a way that adds up peaks at the exact same energy together to obtain their counts) and not for proper evaluation of their
    """
    def __init__(self, spectrum, isotope_list):
        self.signatures = []
        for isotope in isotope_list:
            last_iso = isotope.split(arrow)[-1]
            selected_spectrum = [line for line in spectrum if isotope in line.origin.keys()]
            sig = GammaSignature(selected_spectrum, last_iso)
            
            # dynamically update the list of matching index. This is bad programming and I'm sorry. I typed this up on a bus on the way home from an experiment.
            matching_indices = [ind for ind, sig_old in enumerate(self.signatures) if sig_old.isotope==last_iso]
            # add it to the self.signature.
            if matching_indices:
                self.signatures[matching_indices[0]] += sig
            else:
                self.signatures.append(sig)

# TODO: each line could've had a list as an attr called .parent = [] so that we know what that line originated from?