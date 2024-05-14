import sys, logging, importlib

def main():

    # MAYBE.. should change legacy file names to match callable module names
    module_map = {\
        "filter_by_density": "filter_poly", \
        # "filter_by_density_v1": "filter_density", \
        "filter_by_boundary": "filter_boundary", \
        "make_spatial_minibatch": "make_spatial_minibatch",\
        "make_dge": "make_dge_univ", \
        "make_sge_by_hexagon": "make_sge_by_hexagon", \
        "fit_model": "init_model_selection", \
        # "lda": "lda_univ", \
        "transform": "transform_univ", \
        "choose_color": "choose_color", \
        "plot_base": "plot_base", \
        "de_bulk": "de_bulk", \
        "factor_report": "factor_report", \
        "slda_decode": "slda_decode", \
        "plot_base": "plot_base", \
        "plot_hexagon": "plot_hexagon", \
        "plot_pixel_multi": "plot_pixel_multi", \
        "plot_pixel_full": "plot_pixel_full", \
        "plot_pixel_single": "plot_pixel_single", \
        "run_together": "run_together", \
    }

    if len(sys.argv) < 2:
        print("Usage: ficture <command> <args>, ficture <command> -h to see arguments for each command")
        print("Available commands:\n"+"\t".join(list(module_map.keys()) ))
        return
    elif sys.argv[1] not in module_map:
        print("Unknown command: "+sys.argv[1])
        print("Available commands:\n"+"\t".join(list(module_map.keys()) ))
        return

    function_name = sys.argv[1]
    module_name = "ficture.scripts." + module_map[function_name]
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    function(sys.argv[2:])
