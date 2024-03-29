"""
Module created to read in fispact input file format that specifies the irradiation schedule.
"""
import os
from collections import namedtuple as _named_tuple
from numpy import cumsum as _cumulative_sum
from foilselector.simulation.efficiency import EfficiencyCurve

def parse_fispact_input_text(text_block):
    """
    Convert a file describing the irradiation, cooling and measurement schedule
    written in FISPACT input file format into a list of str.

    This is done with the following steps:
    Remove comment cards, and merge delimiters. 
    Then return the text as a list of strings (each of which contain a keyword or a parameter value as str).
    Note that the keywords are returned in uppercase value as the syntax for fispact input file format is case-insensitive.
    """
    # remove the comment cards:
    bracket_level = 0
    keywords_and_params = ""

    # escape comments.
    for char in text_block:
        if char=="<":
            bracket_level += 1
        elif char==">":
            bracket_level -= 1
        else:
            if bracket_level==0:
                keywords_and_params += char

    # merge delimiters
    keywords_and_params = [word for word in keywords_and_params.split() if len(word)>0]
    return keywords_and_params

unit_conversion = {
    "SECS":1,
    "MINS":60,
    "HOURS":3600,
    "DAYS":86400,
    "YEARS":365.25*86400
}

end_step_kw = ("STEP", "SPECTRUM", "ATOMS") # keyword to indicate that irradiation has ended for this phase
gamma_acquisition_kw = ("SPECTRUM", "ATOMS")

class Step():
    """a base class defining an Irradiation or cooling step"""
    def __init__(self, duration, flux):
        """f"""
        assert flux>=0, "Accepts non-negative flux only"
        self.duration = duration # the duration where this flux is held for.
        self.flux = flux

    def __str__(self):
        return "< {} lasting {}s >".format(self.__class__.__name__, self.duration)

class IrradiationStep(Step):
    """A step involving non-zero flux irradiation, and no gamma acquision."""
    def __init__(self, duration, flux):
        assert flux>0.0, "Use {} instead if flux==0.0".format(CoolingStep)
        super().__init__(duration, flux)

    def __str__(self):
        return "< {} of flux={}cm^2 s^-1 lasting {}s >".format(self.__class__.__name__, self.flux, self.duration)

class CoolingStep(Step):
    def __init__(self, duration):
        super(CoolingStep, self).__init__(duration, 0.0)

class GammaSpectrometryStep(CoolingStep):
    def __init__(self, duration):
        super(GammaSpectrometryStep, self).__init__(duration)

times_and_flux = _named_tuple("times_and_flux", ["fluence", "flux", "a", "b", "c"])

class Schedule():
    # a collection of steps
    def __init__(self, *step_list):
        durations, fluxes = [0.0], []
        for step in step_list:
            durations.append(step.duration)
            fluxes.append(step.flux)

        self.step_start_times = _cumulative_sum(durations)[:-1]
        self.step_end_times = _cumulative_sum(durations)[1:]
        self.fluxes = fluxes
        self.is_irradiation = [isinstance(step, IrradiationStep) for step in step_list]
        self.is_gamma_measurement = [isinstance(step, GammaSpectrometryStep) for step in step_list]

    def measurable_irradiation_effects(self):
        # number of gamma measurement steps = sum(self.is_gamma_measurement)
        times_list = [] # known as time a, time b and time c.
        # double for-loop: one to loop over each measurement step, and one for each measurement step.
        for curr_step_num in range(len(self.is_gamma_measurement)):
            if self.is_gamma_measurement[curr_step_num]:
                for prev_step_num in range(len(self.is_irradiation[:curr_step_num+1])):
                    if self.is_irradiation[prev_step_num]:
                        zero = self.step_start_times[prev_step_num]
                        a = self.step_end_times[prev_step_num] - zero
                        flux = self.fluxes[prev_step_num]
                        fluence = self.fluxes[prev_step_num] * a

                        b = self.step_start_times[curr_step_num] - zero
                        c = self.step_end_times[curr_step_num] - zero

                        times_list.append(times_and_flux(fluence, flux, a, b, c))
        return times_list

    def copy(self):
        new_obj = type(self).__new__(self.__class__)
        new_obj.__dict__.update(self.__dict__)
        return new_obj

    def __truediv__(self, denominator):
        """div is implemented to reduce the number of denominator."""
        new_obj = self.copy()
        new_obj.fluxes = [f/denominator for f in new_obj.fluxes]
        return new_obj

def process_one_material_schedule_text(keywords_and_params):
    """
    Turn the list of keywords and parameters str as read from the FISPACT-format input file
    into a Schedule object.

    2022-10-31 00:08:56 Sorry, I don't have time to add a syntax manual here yet. You'll have to figure it out from the code for now.
    """
    COOLING_ONLY = False
    steps_schedule = [] # empty container to contain all steps for one material

    keywords_and_params = [word for word in keywords_and_params] # make a copy

    while len(keywords_and_params)>0:
        keyword = keywords_and_params.pop(0)

        if keyword.upper()=="FLUX":
            assert not COOLING_ONLY, "Only allowed to change flux before the ZERO keyword."
            flux = float(keywords_and_params.pop(0))

        elif keyword.upper()=="TIME":
            numerical_duration = float(keywords_and_params.pop(0)) # expects a parameter

            next_word_probably_unit = (keywords_and_params.pop(0)) # expects a keyword which is a time unit.
            try:
                unit = next_word_probably_unit.upper()
                duration_seconds = numerical_duration * unit_conversion[unit] # raises KeyError if next_word_probably_unit isn't a time unit, and skips the following line.
                time_end_kw = keywords_and_params.pop(0).upper() # expects a keyword
            except KeyError: # if the user chose to use the default unit of seconds, therefore the line above ^ isn't executed
                # next_word_probably_unit 
                time_end_kw = next_word_probably_unit.upper()
                duration_seconds = numerical_duration * unit_conversion['SECS']

            assert time_end_kw in end_step_kw, f"Must end this irradiation step in one of the pre-approved keywords, not {time_end_kw}, {len(keywords_and_params)}"
            if time_end_kw.upper() in gamma_acquisition_kw:
                steps_schedule.append(GammaSpectrometryStep(duration_seconds))
                file_path = keywords_and_params.pop(0)
                filepath_assert_err_msg = "Gamma spectrum acquisition keyword must be followed by a valid file path stating the efficiency curve file location."
                filepath_assert_err_msg += f"There mustn't be any spaces in the file path. Instead we got {file_path}"
                assert os.path.exists(file_path), filepath_assert_err_msg
                eff_curve = EfficiencyCurve.from_file(file_path)
            else: # if only a STEP keyword is used:
                if flux>0.0:
                    steps_schedule.append(IrradiationStep(duration_seconds, flux))
                else:
                    steps_schedule.append(CoolingStep(duration_seconds))
                eff_curve = None
                
        elif keyword.upper()=="ZERO": # only exist for fispact compatibility, I suppose.
            # an optional keyword suggesting that there should be zero flux after this point
            COOLING_ONLY = True
            assert flux==0.0, "Must set flux to 0 before the ZERO keyword."

    return steps_schedule, eff_curve

def cut_text_at_sample(full_control_text):
    """
    Cut the list of keywords and parameters into lists of strings,
    where each string is a keyword/parameter.
    The order of str's in each word follows the order of the passage.
    A new list is started whenever a "SAMPLE (sample_name)" pair is detected.

    Returns
    -------
    A dictionary with the sample_name as key and word list as values.
    The two keywords "SAMPLE (sample_name)" are already removed from these lists.
    """
    material_names, material_schedule_text = [], [] # container for the material names and the main text describing the material irradiation schedules.
    if full_control_text[0].upper()!="SAMPLE":
        material_names.append("unknown_sample")
        material_schedule_text.append([])
        print("No material names found, using 'unknown_sample' as the default name.")

    while len(full_control_text)>0:
        new_word = full_control_text.pop(0)
        if new_word.upper()=="SAMPLE":
            material_names.append(full_control_text.pop(0))
            material_schedule_text.append([])
            # create a new list
        else:
            material_schedule_text[-1].append(new_word)
    return dict(zip(material_names, material_schedule_text))

def read_fispact_irradiation_schedule(schedule_text_block):
    """
    Different samples ares separated using the SAMPLE keyword.
    Otherwise the rest of the irradiation schedule follows the fispact input format:
    use "FLUX number" to define the flux of this step;
    use "TIME value SECS/MINS/HOURS/DAYS/YEARS" to define the duration of this step;
    use "STEP/SPECTRUM/ATOM" to define the end of a step, and whether step involves gamma acquisition (SPECTRUM/ATOM) or not (STEP).

    Parameters
    ----------
    schedule_text_block: a chunk of raw text (read from a text file) in the syntax stated above.
    """
    full_control_text = parse_fispact_input_text(schedule_text_block)

    word_lists_sorted_by_sample = cut_text_at_sample(full_control_text)

    schedule_dict, efficiency_dict = {}, {}
    for name, texts in word_lists_sorted_by_sample.items():
        schedule, efficiency_curve = process_one_material_schedule_text(texts)
        schedule_dict[name] = Schedule(*schedule)
        efficiency_dict[name] = efficiency_curve
        # efficiency_curve.plot()
    return schedule_dict, efficiency_dict
