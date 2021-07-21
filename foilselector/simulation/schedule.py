"""
Module created to read in fispact input file format that specifies the irradiation schedule.
"""
from collections import namedtuple as _named_tuple
from numpy import cumsum as _cumulative_sum

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

    for char in text_block:
        if char=="<":
            bracket_level += 1
        elif char==">":
            bracket_level -= 1
        else:
            if bracket_level==0:
                keywords_and_params += char

    # merge delimiters
    keywords_and_params = [word.upper() for word in keywords_and_params.split() if len(word)>0]
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

times_and_flux = _named_tuple("times_and_flux", ["fluence", "a", "b", "c"])

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

    def irradiation_sum_times(self):
        # number of gamma measurement steps = sum(self.is_gamma_measurement)
        times_list = [] # known as time a, time b and time c.
        for curr_step_num in range(len(self.is_gamma_measurement)):
            if self.is_gamma_measurement[curr_step_num]:
                for prev_step_num in range(len(self.is_irradiation[:curr_step_num+1])):
                    if self.is_irradiation[prev_step_num]:
                        zero = self.step_start_times[prev_step_num]
                        a = self.step_end_times[prev_step_num] - zero
                        flux = self.fluxes[prev_step_num]

                        b = self.step_start_times[curr_step_num] - zero
                        c = self.step_end_times[curr_step_num] - zero

                        times_list.append(times_and_flux(flux*a, a, b, c))
        return times_list

def keyword_and_param_reader(keywords_and_params):
    """
    Turn the list of keywords and parameters str as read from the FISPACT-format input file
    into a Schedule object.
    """
    COOLING_ONLY = False
    steps_schedule = [] # empty container to contain all steps

    while len(keywords_and_params)>0:
        keyword = keywords_and_params.pop(0)
        if keyword=="FLUX":
            assert not COOLING_ONLY, "Only allowed to change flux before the ZERO keyword."
            flux = float(keywords_and_params.pop(0))

        elif keyword=="TIME":
            numerical_duration = float(keywords_and_params.pop(0)) # expects a parameter

            units = keywords_and_params.pop(0) # expects a keyword
            duration_seconds = numerical_duration * unit_conversion[units]

            time_end_kw = keywords_and_params.pop(0) # expects a keyword
            assert time_end_kw in end_step_kw, "Must end this irradiation step in one of the pre-approved keywords."
            if time_end_kw in gamma_acquisition_kw:
                steps_schedule.append(GammaSpectrometryStep(duration_seconds))
            else: # if only a STEP keyword is used:
                if flux>0.0:
                    steps_schedule.append(IrradiationStep(duration_seconds, flux))
                else:
                    steps_schedule.append(CoolingStep(duration_seconds))

        elif keyword=="ZERO":
            # an optional keyword suggesting that there should be zero flux after this point
            COOLING_ONLY = True
            assert flux==0.0, "Must set flux to 0 before the ZERO keyword."

    return steps_schedule

def read_fispact_irradiation_schedule(schedule_text_block):
    keywords_and_params = parse_fispact_input_text(schedule_text_block)
    steps_schedule = keyword_and_param_reader(keywords_and_params)
    return Schedule(*steps_schedule)