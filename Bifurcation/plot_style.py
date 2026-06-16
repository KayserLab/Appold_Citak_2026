import matplotlib.pyplot as plt


def configure_plot_style():
    style_config = {
        "font.size": 7,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
    plt.rcParams.update(style_config)
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
