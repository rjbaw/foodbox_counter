import argparse
from pyModbusTCP.server import ModbusServer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-address', type=str, default='localhost', help='server address')
    parser.add_argument('-p', type=int, default=502, help='TCP port')
    args = parser.parse_args()
    server = ModbusServer(host=args.address, port=args.p) # no_block=True
    server.start()
