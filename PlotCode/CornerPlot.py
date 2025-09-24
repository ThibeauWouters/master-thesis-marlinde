import numpy as np
import corner as corn #🌽
import h5py
import matplotlib.pyplot as plt
import json
from scipy.stats import norm, mode
from tqdm import tqdm  


default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        #color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)


param_to_latex = {
    "chirp_mass": r"\mathcal{M}",
    "mass_ratio": r"q",
    "a_1": r"a_1",
    "a_2": r"a_2",
    "tilt_1": r"\theta_1",
    "tilt_2": r"\theta_2",
    "phi_12": r"\phi_{12}",
    "phi_jl": r"\phi_{JL}",
    "lambda_1": r"\Lambda_1",
    "lambda_2": r"\Lambda_2",
    "luminosity_distance": r"d_L",
    "geocent_time": r"t_c",
    "dec": r"\delta",
    "ra": r"\alpha",
    "theta_jn": r"\theta_{JN}",
    "psi": r"\psi",
    "phase": r"\phi"
}

all_params=['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'lambda_1', 'lambda_2', 'luminosity_distance', 'geocent_time', 'dec', 'ra', 'theta_jn', 'psi', 'phase']


def sigma_to_levels(sigma):
    """Return corner.py 'levels' array for 1..sigma sigma regions."""
    return [norm.cdf(s) - norm.cdf(-s) for s in range(1, sigma+1)]


def make_cornerplot(data, parameters, truth=False, median = False, average=False, mode=False, color="blue", sigma=4):
    """
    data[.h5py file]: data for the corner plot
    parameters[list]: list of parameters to plot
    truth[Bool]: plot injected values when set to True
    median[Bool]: plot median values of the posterior when set to True
    average[Bool]: plot the average values of the posterior when set to True
    mode[Bool]: plot the mode values of the posterior when set to True 
    color[str]: color for the corner plots
    sigma[int]: number of sigma regions in the corner plots (max. 8)

    Returns a dictionary containing the averages of the posterior 
    """
    with h5py.File(data, 'r') as f:
        
        #For general info, print ln bayes factor and sampling time
        log_bayes_factor = f["log_bayes_factor"][()]
        print(f"Log Bayes factor: {log_bayes_factor}")
        
        sampling_time = f["sampling_time"][()]
        sampling_time_hrs = sampling_time / 3600.0
        print(f"Sampling time: {sampling_time_hrs:.2f} hours")
        
        #Load the priors used
        prior_keys_to_skip = ['__prior_dict__', '__module__', '__name__']
        priors_bytes = f["priors"][()]        # bytes
        priors_str = priors_bytes.decode()    # bytes → string
        priors_dict = json.loads(priors_str)  # string → dict
        priors_dict = {k: v for k, v in priors_dict.items() if k not in prior_keys_to_skip and "recalib" not in k}
        prior_keys = list(priors_dict.keys())
        #print(f"Prior dict keys: {prior_keys}")
        
        #Load the posterior samples of the prior dict keys, and any other keys added by user
        posterior = f["posterior"]
        posterior_samples = np.array([posterior[key][()] for key in parameters]).T
        #Compute the mean value of each parameter
        param_means = posterior_samples.mean(axis=0)
        #Convert labels to LaTeX format
        latex_labels = [f"${param_to_latex.get(p, p)}$" for p in parameters]
        param_means_dict = {}
        for i in range(len(parameters)):
             param_means_dict.update({parameters[i] : param_means[i]})

    if average == True and truth == True or average == True and mode == True or mode == True and truth == True:
         print("Cannot both plot average and truth! Plotting truth values.")  #aanpassen 
         average = False    

    try:
        print(f"Creating corner plot . . .")
        if sigma > 8:
                print("Too many sigma's, no bueno! Setting sigma to 8")
                sigma=8
        levels = sigma_to_levels(sigma)

        truths = None
        quantiles = None

        if average:
             truths = param_means
             print("Plotting the averages of the posteriors")
             print("Averages of the posterior: ")
             print(param_means_dict)
        
        if truth:
            if "injection_parameters" in f.keys():
                injection = f["injection_parameters"]
                truths = np.array([injection[key][()] for key in parameters])
                print("Plotting the injected values")
            else:
                print("No injected values found in file!")
        
        if median:
             quantiles=[0.5]
             print("Plotting the medians of the posterior")

        if mode:
            param_modes = []
            param_modes_dict = {}
            for i in range(posterior_samples.shape[1]):
                hist, bin_edges = np.histogram(posterior_samples[:, i], bins=100)
                max_bin_index = np.argmax(hist)
                mode_val = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index+1])
                param_modes.append(mode_val)
                param_modes_dict.update({parameters[i] : mode_val})
            print("Modes of the posterior: ")
            print(param_modes_dict)


            param_modes = np.array(param_modes)
            truths = param_modes
            print("Plotting the modes of the posterior")
        

        corn.corner(posterior_samples,
                        labels=latex_labels,
                        color=color,
                        levels=levels,
                        truths=truths,
                        quantiles=quantiles,
                        **default_corner_kwargs)
        

    except Exception as e:
        print("Something went wrong!")
        return 