import dearpygui.dearpygui as dpg
import taichi as ti
from taichi.math import vec4
import numpy as np
import time

dpg.create_context()

ti.init(arch = ti.cuda)
field = ti.Vector.field(4, float, (800, 800))

@ti.kernel
def get_random_field():
    for i, j in field:
        field[i, j] = vec4([ti.random(float), ti.random(float), ti.random(float), ti.random(float)])

def button_callback(sender):
    global mode
    mode = sender

raw_data = np.random.uniform(0, 1, (800, 800, 4))

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(width=800, height=800, default_value=raw_data, format=dpg.mvFormat_Float_rgba, tag="texture_tag")

def toggle_circle():
    disp = dpg.get_item_configuration("circle")["show"]
    dpg.configure_item("circle", show = not disp)
    print(dpg.get_value("val_scale"))

with dpg.window(label="Tutorial", tag = "disp", no_bring_to_front_on_focus = True):
    dpg.add_image("texture_tag")
    dpg.draw_circle((100, 100), 10, color = (255, 255, 0, 255), fill = (255, 255, 0), tag=f"circle")

def value_sync(set_tag: str):
    def value_sync_inner(_sender, app_data, _user_data):
        dpg.set_value(set_tag, app_data)
    return value_sync_inner

def create_slider(label: str, tag: str, min_v: float, max_v: float, default: float):
    with dpg.group(horizontal = True):
        dpg.add_input_float(tag = f"{tag}_input", default_value = default, 
                                width = 110, callback = value_sync(tag))
        dpg.add_slider_float(label = label, tag = tag, min_value = min_v,
                            max_value = max_v, default_value = default, width = 120, callback = value_sync(f"{tag}_input"))

def change_text(sender, app_data):
    if app_data == 27:
        dpg.stop_dearpygui()
    print(f"Key Button: {app_data}")

if __name__ == "__main__":
    mode = "da_only"
    with dpg.handler_registry():
        dpg.add_key_release_handler(callback=change_text)

    with dpg.window(label="Control", tag = "primary_control"):
        dpg.window()
        dpg.add_button(label = "Toggle circle", callback = toggle_circle)
        create_slider("Value scale", "v_scale", -3, 10, 1)
        create_slider("Max time", "max_time", 0.5, 10, 2)
        create_slider("Time", "time", 0.0, 2, 1)
        create_slider("Emitter position", "emitter_pos", 0.05, 5.0, 1)
        create_slider("Canvas scale", "scale", 10.0, 1000.0, 10)
        create_slider("Sigma A", "ua", 0.01, 1.0, 0.02)
        create_slider("Sigma S", "us", 5.0, 200.0, 10)
        with dpg.group(horizontal = True):
            dpg.add_checkbox(label = 'Distance sampling', tag = 'dist_sample', default_value = False)
            dpg.add_checkbox(label = 'Use Tr', tag = 'use_tr', default_value = True)
        with dpg.group(horizontal = True):
            dpg.add_button(label = 'DA only', tag = 'da_only', width = 100, callback = button_callback)
            dpg.add_button(label = 'DA Tr', tag = 'da_tr', width = 100, callback = button_callback)

    dpg.create_viewport(title='Custom Title', width=800, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    cnt = 0
    start_time = time.time()
    print("View port created")
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        get_random_field()
        raw_data = field.to_numpy()
        dpg.configure_item("circle", center = (dpg.get_value("emitter_pos") * dpg.get_value("scale"), 400))
        dpg.configure_item("texture_tag", default_value=raw_data)
        cnt += 1
    end_time = time.time()
    print(f"FPS: {cnt / (end_time - start_time)}")
    dpg.destroy_context()