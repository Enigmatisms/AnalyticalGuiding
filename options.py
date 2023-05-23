import configargparse

__all__ = ["get_options"]

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("-b", "--backward",
                                     default = False, action = "store_true", help = "Whether to go backwards when RTS")

    parser.add_argument("--ua",      default = 0.02, help = "absorption coeff", type = float)
    parser.add_argument("--us",      default = 10.0, help = "scattering coeff", type = float)

    parser.add_argument("--xmin",    default = 0, help = "Minimum x", type = float)
    parser.add_argument("--xmax",    default = 0.95, help = "Maximum x", type = float)
    parser.add_argument("--x_point", default = 0.5, help = "Point x", type = float)

    parser.add_argument("--eps",     default = 1.0, help = "Position of the emitter", type = float)
    parser.add_argument("--tmin",    default = 0.5, help = "Minimum t", type = float)
    parser.add_argument("--tmax",    default = 1.5, help = "Maximum t", type = float)
    parser.add_argument("--t_point", default = 1.0, help = "Point t", type = float)
    
    # 注意此处，t plus 的意思是在原始剩余时间上增加的时间。原始剩余时间（RT）根据当前路径到光源入射t = 0点的最短路去算
    # 假设我们希望 sample 别的时间点（大于最短路的时间点）, 则可以设置此时间

    parser.add_argument("--t_plus_num",  default = 2, help = "Number of plus-time to compare", type = int)
    parser.add_argument("--t_plus_val",  default = 0.5, help = "Interval between added time", type = float)

    parser.add_argument("--pnum",    default = 1000, help = "Number of points to draw", type = int)
    parser.add_argument("--snum",    default = 5000, help = "Number of samples", type = int)
    
    parser.add_argument("--mode",    default = 'rts', choices=['time', 'space', 'rts'], help = "Visualization mode", type = str)
    parser.add_argument("--func",    default = 'h_n', choices=['h_n', 'h_d', 'full'], help = "Solution to use", type = str)
    parser.add_argument("--sol",     default = "unit", choices=['unit', 'physical'], help = "Speed of light: 1 or physical", type = str)

    if delayed_parse:
        return parser
    return parser.parse_args()