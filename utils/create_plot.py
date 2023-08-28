""" This class is for creating DPG plotting utilities
    @author: Qianyue He
    @date: 2023-5-25
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import List, Union

class PlotTools:
    tag_cnt = 0
    series_cnt = 0
    cursor_cnt = 0
    series_map = {"bar": dpg.add_bar_series, "line": dpg.add_line_series}

    @staticmethod
    def make_plot(
        width:int, height:int, title: str, labels: List[str], 
        val_nums: Union[int, List[int]] = None, all_init_xs: List[np.ndarray] = None,
        mode: str = "line", xy_labels = ['x', 'y'], use_cursor = False
    ):
        if val_nums is None and all_init_xs is None:
            raise ValueError(f"No initialization while trying to create plot labelled '{title}'")
        series_func = PlotTools.series_map[mode]
        with dpg.plot(label = title, height = height, width = width, anti_aliased = True):
            # optionally create legend
            dpg.add_plot_legend()

            # REQUIRED: create x and y axes
            x_axis_tag = f"x_axis_{PlotTools.tag_cnt}"
            y_axis_tag = f"y_axis_{PlotTools.tag_cnt}"
            dpg.add_plot_axis(dpg.mvXAxis, label=xy_labels[0], tag = x_axis_tag)
            dpg.add_plot_axis(dpg.mvYAxis, label=xy_labels[1], tag = y_axis_tag)
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
                                label=label, parent = y_axis_tag, tag = f"series_tag_{PlotTools.series_cnt}", **kwargs)
                        PlotTools.series_cnt += 1
                else:
                    for val_num, label in zip(val_nums, labels):
                        xs = np.linspace(0, 1, val_num)
                        ys = np.zeros_like(xs)
                        if mode == "bar":
                            kwargs["weight"] = 1 / val_num
                        series_func(xs, ys, 
                                label=label, parent = y_axis_tag, tag = f"series_tag_{PlotTools.series_cnt}", **kwargs)
                        PlotTools.series_cnt += 1
            else:
                for init_xs, label in zip(all_init_xs, labels):
                    ys = np.zeros_like(init_xs)
                    if mode == "bar":
                        kwargs["weight"] = init_xs[1] - init_xs[0]
                    series_func(init_xs, ys, 
                                label = label, parent = y_axis_tag, tag = f"series_tag_{PlotTools.series_cnt}", **kwargs)
                    PlotTools.series_cnt += 1
            if use_cursor:
                dpg.add_line_series([0, 0], [0, 1], label = "cursor", parent = y_axis_tag, tag = f"cursor_{PlotTools.cursor_cnt}")
                PlotTools.cursor_cnt += 1
        PlotTools.tag_cnt += 1

    # ====================== Draw time ellipse ======================

    @staticmethod
    def get_ellipse(f1_x: float, f2_x: float, y_pos: float, a2: float):
        """ Creating a ellipse, a2 is target time (distance to two foci)
            Note that inputs are scaled (to window coordinates)
        """
        center_x = (f1_x + f2_x) / 2.
        fc = (f2_x - f1_x) / 2.
        long_axis = a2 / 2.
        short_axis = np.sqrt((long_axis) ** 2 - (fc ** 2))
        pmin = (center_x - long_axis, y_pos - short_axis)
        pmax = (center_x + long_axis, y_pos + short_axis)
        return pmin, pmax

    @staticmethod
    def create_ellipse(f1_x: float, f2_x: float, y_pos: float, a2: float, color = (0, 255, 0, 100), tag = "time_ellipse"):
        """ Create dearpygui ellipse item 
            Note that inputs are scaled (to window coordinates)
        """
        pmin, pmax = PlotTools.get_ellipse(f1_x, f2_x, y_pos, a2)
        dpg.draw_ellipse(pmin, pmax, tag = tag, color = color, show = False, thickness = 2, segments = 128)

    @staticmethod
    def toggle_ellipse(f1_x: float, f2_x: float, y_pos: float, a2: float, tag = "time_ellipse"):
        """ Modify dearpygui ellipse item 
            Note that inputs are scaled (to window coordinates)
        """
        pmin, pmax = PlotTools.get_ellipse(f1_x, f2_x, y_pos, a2)
        dpg.configure_item(tag, pmin = pmin)
        dpg.configure_item(tag, pmax = pmax)

    # ====================== Draw peak path ======================

    @staticmethod
    def create_peak_plots(f1_x: float, f2_x: float, y_pos: float, scale: float):
        """ Creating peak plots 
            Note that inputs are NOT scaled (still in world frame, except y_pos)
        """
        px = (f1_x + f2_x) / 2.
        length1 = np.sqrt((px - f1_x) ** 2)       # vertex to peak distance
        length2 = np.sqrt((px - f2_x) ** 2)       # emitter to peak distance
        
        p_peak    = (px * scale, y_pos)
        p_vertex  = (f1_x * scale, y_pos)
        p_emitter = (f2_x * scale, y_pos)
        center1   = (0.5 * (px + f1_x) * scale - 40, y_pos - 30)
        center2   = (0.5 * (px + f2_x) * scale - 40, y_pos - 30)

        dpg.draw_line(p1 = p_vertex, p2 = p_peak, color = (255, 255, 0, 100), tag = "peak_line1", show = False)
        dpg.draw_line(p1 = p_peak, p2 = p_emitter, color = (255, 255, 0, 100), tag = "peak_line2", show = False)
        dpg.draw_circle(center = p_peak, radius = 5, color = (255, 255, 0, 100), tag = "peak_point", show = False)
        dpg.draw_text(pos = center1, text = f"{length1:4f}", color = (255, 255, 255, 200), 
                      tag = "peak_text1", show = False, size = 16)
        dpg.draw_text(pos = center2, text = f"{length2:4f}", color = (255, 255, 255, 200), 
                      tag = "peak_text2", show = False, size = 16)

    @staticmethod
    def toggle_peak_plots(px: float, py: float, f1_x: float, f2_x: float, y_pos: float, scale: float):
        """ Modify peak plots 
            Note that inputs are NOT scaled (still in world frame, except y_pos)
        """
        world_y = y_pos / scale
        length1 = np.sqrt((px - f1_x) ** 2 + (py - world_y) ** 2)       # vertex to peak distance
        length2 = np.sqrt((px - f2_x) ** 2 + (py - world_y) ** 2)       # emitter to peak distance
        
        p_peak    = (px * scale, py * scale)
        p_vertex  = (f1_x * scale, y_pos)
        p_emitter = (f2_x * scale, y_pos)
        center_y  = (py + world_y) * 0.5 
        center1   = (0.5 * (px + f1_x) * scale - 40, center_y * scale - 30)
        center2   = (0.5 * (px + f2_x) * scale - 40, center_y * scale - 30)

        dpg.configure_item("peak_line1", p1 = p_vertex, p2 = p_peak)
        dpg.configure_item("peak_line2", p1 = p_peak, p2 = p_emitter)
        dpg.configure_item("peak_point", center = p_peak)
        dpg.configure_item("peak_text1", pos = center1, text = f"{length1:4f}")
        dpg.configure_item("peak_text2", pos = center2, text = f"{length2:4f}")
