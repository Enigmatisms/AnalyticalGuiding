import dearpygui.dearpygui as dpg
import numpy as np
from typing import List, Union

class PlotTools:
    tag_cnt = 0
    series_cnt = 0
    series_map = {"bar": dpg.add_bar_series, "line": dpg.add_line_series}

    @staticmethod
    def make_plot(
        width:int, height:int, title: str, labels: List[str], 
        val_nums: Union[int, List[int]] = None, all_init_xs: List[np.ndarray] = None,
        mode: str = "line",
    ):
        if val_nums is None and all_init_xs is None:
            raise ValueError(f"No initialization while trying to create plot labelled '{title}'")
        series_func = PlotTools.series_map[mode]
        with dpg.plot(label = title, height = height, width = width):
            # optionally create legend
            dpg.add_plot_legend()

            # REQUIRED: create x and y axes
            axis_tag = f"y_axis_{PlotTools.tag_cnt}"
            dpg.add_plot_axis(dpg.mvXAxis, label="x")
            dpg.add_plot_axis(dpg.mvYAxis, label="y", tag = axis_tag)
            kwargs = {}
            # series belong to a y axis
            if all_init_xs is None:
                if type(val_nums) is int:
                    for label in labels:
                        xs = np.linspace(0, 1, val_nums)
                        ys = np.zeros_like(xs)
                        if mode == "bar":
                            kwargs["weight"] = 1 / val_nums
                        series_func(xs, ys,
                                label=label, parent = axis_tag, tag = f"series_tag_{PlotTools.series_cnt}", **kwargs)
                        PlotTools.series_cnt += 1
                else:
                    for val_num, label in zip(val_nums, labels):
                        xs = np.linspace(0, 1, val_num)
                        ys = np.zeros_like(xs)
                        if mode == "bar":
                            kwargs["weight"] = 1 / val_num
                        series_func(xs, ys, 
                                label=label, parent = axis_tag, tag = f"series_tag_{PlotTools.series_cnt}", **kwargs)
                        PlotTools.series_cnt += 1
            else:
                for init_xs, label in zip(all_init_xs, labels):
                    ys = np.zeros_like(init_xs)
                    if mode == "bar":
                        kwargs["weight"] = init_xs[1] - init_xs[0]
                    series_func(init_xs, ys, 
                                label = label, parent = axis_tag, tag = f"series_tag_{PlotTools.series_cnt}", **kwargs)
                    PlotTools.series_cnt += 1
        PlotTools.tag_cnt += 1