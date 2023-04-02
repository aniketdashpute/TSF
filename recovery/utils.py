import configargparse
def config_parser(path):
    parser = configargparse.ArgParser(default_config_files=[path])
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    
    # Capture data information
    parser.add_argument("--strCommon", type=str, default='',
                        help='prefix to experiment name')
    parser.add_argument("--Nz", type=int, default=50,
                        help='layers in depth direction')
    parser.add_argument("--total_time_captured", type=float, default=60,
                        help='time in seconds')
    parser.add_argument("--total_time", type=float, default=60,
                        help='time in seconds')
    parser.add_argument("--numCycles", type=int, default=1,
                        help='how many cycles of on/off?')
    parser.add_argument("--timeCycleON", type=int, default=30,
                        help='How many seconds for each cycle is source ON?')
    parser.add_argument("--time_ON", type=int, default=30,
                        help='seconds')
    parser.add_argument("--delX", type=float, default=0.5e-3,
                        help='mm')
    parser.add_argument("--delY", type=float, default=0.5e-3,
                        help='mm')
    parser.add_argument("--delZ", type=float, default=0.5e-3,
                        help='mm')
    
    # FD parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--total_iterations", type=int, default=400,
                        help='total iterations')
    parser.add_argument("--numLayersK", type=int, default=1,
                        help='total layers of K')
    parser.add_argument("--numLayersEps", type=int, default=1,
                        help='total layers of Eps')
    parser.add_argument("--EpsFactor", type=float, default=1.0,
                        help='Eps factor')
    parser.add_argument("--delTimgf", type=int, default=1,
                        help='delta T factor')
    parser.add_argument("--tremove", type=float, default=0.0,
                        help='remove some initial period of scan')

    return parser