import os
import time
import argparse
from pyModbusTCP.client import ModbusClient
import subprocess
import curses
import time
#from counter import Detect

def main(stdscr):
    k = 0
    stdscr.clear()
    stdscr.refresh()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)
    stdscr.keypad(True)
#    stdscr.nodelay(1)
    curses.halfdelay(1)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)
    client_started=False
    server_started=False
    server_status = "SERVER NOT RUNNING"
    client_status = "CLIENT NOT RUNNING"
    status = "WAITING"
    initial = True
    c = ModbusClient(host=args.address,
                     port=502,
                     unit_id=args.i,
                     auto_open=True,
                     auto_close=True)

    while True:
        stdscr.clear()
        start_y = 1
        stdscr.attron(curses.color_pair(3))
        height, width = stdscr.getmaxyx()
        whstr = "Width: {}, Height: {}".format(width, height)
        stdscr.addstr(0, 0, " " *(width-len(whstr)))
        stdscr.addstr(0, int(width-len(whstr)-1), whstr)
#        stdscr.addstr(0, 0, whstr, curses.color_pair(1))
        statusbarstr = "Press 'q' to exit | STATUS: {}".format(status)
        try:
            stdscr.addstr(height-1, 0, statusbarstr)
            stdscr.addstr(height-1, len(statusbarstr), " " * (width - len(statusbarstr) - 1))
        except Exception as e:
            print(e)
            print("Increase window size!")
        stdscr.attroff(curses.color_pair(3))

        title = "MODBUS TCP SERVER"[:width-1]
        start_x_title = 1
        stdscr.attron(curses.color_pair(2))
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(start_y, start_x_title, title)
        stdscr.attroff(curses.color_pair(2))
        stdscr.attroff(curses.A_BOLD)

        stdscr.attron(curses.A_BOLD)
        keystr = "SLAVE ID: {0}".format(args.i)[:width-1]
        stdscr.addstr(start_y, width-len(keystr)-1, keystr)
        dis_address = "ADDRESS: {0}:{1}".format(args.address,502)[:width-1]
        stdscr.addstr(start_y+3, start_x_title, dis_address)
        stdscr.addstr(start_y+5, start_x_title+1, "SLAVE ADDRESS:")
        stdscr.attroff(curses.A_BOLD)

        subtitle = "raveeroj.baw@cpf.co.th"[:width-1]
        start_x_subtitle = 2
        stdscr.addstr(start_y+1, start_x_subtitle, subtitle, curses.color_pair(1))
        stdscr.addstr(start_y+1, width-len(server_status)-1, server_status)
        stdscr.addstr(start_y+2, width-len(client_status)-1, client_status)

        stdscr.attron(curses.A_BOLD)
        instructions = "Instructions"
        stdscr.addstr(start_y+5, int(width - len(instructions) - 1), instructions)
        stdscr.addstr(start_y+10, int(width - len("DEBUG") - 1), "DEBUG")
        stdscr.attroff(curses.A_BOLD)
        refreshanykey = "Refresh :Any"
        stdscr.addstr(start_y+6, int(width - len(refreshanykey) - 1), refreshanykey)
        kquit = "Quit :Q"
        stdscr.addstr(start_y+7, int(width - len(kquit) - 1), kquit)
        kstart = "Start All :T"
        stdscr.addstr(start_y+8, int(width - len(kstart) - 1), kstart)
        kstop = "Stop All :Y"
        stdscr.addstr(start_y+9, int(width - len(kstop) - 1), kstop)
        kstarts = "Start Server :S"
        stdscr.addstr(start_y+11, int(width - len(kstarts) - 1), kstarts)
        kstartk = "Kill Server :K"
        stdscr.addstr(start_y+12, int(width - len(kstartk) - 1), kstartk)
        kclient = "Start Client :C"
        stdscr.addstr(start_y+13, int(width - len(kclient) - 1), kclient)
        kclientk = "Kill Client :D"
        stdscr.addstr(start_y+14, int(width - len(kclientk) - 1), kclientk)

        if initial:
            if server_started:
                server_status = "Server running!"
            else:
                bash = "python modbus/modbus_server.py -address {0}".format(args.address)
                server = subprocess.Popen(bash.split(), stdout=subprocess.PIPE)
                server_started = True
                status = "SERVER STARTING"
            if client_started:
                client_status = "Client running!"
            else:
                clients = []
                for i in range(len(stream_txt)):
                    client_shift = stream_index[i].split('-')[0]
                    bash = "python counter.py --slave-id {0} --source {1} --address {2} --client-shift {3}".format(args.i,stream_txt[i],args.address, client_shift)
                    clients.append(subprocess.Popen(bash.split(), stdout=subprocess.DEVNULL))
                timeout = time.time()
                client_started = True
                client_status = "CLIENT RUNNING"
                status = "CLIENT STARTING"
            initial = False

        if not c.is_open():
            if not c.open():
                server_status = "Unable to connnect to "+args.address+":"+str(502)
                status = "WAITING"
        if c.is_open():
            server_status = "SERVER RUNNING"
            if client_started:
                status = "RUNNING"

            start_y, start_x = start_y+7, start_x_title+2
            for i,addr in enumerate(stream_index):
                start, stop = [int(x) for x in addr.split('-')]
                for addr in range(start, stop+1):
                    if clients[i].poll() is not None:
                        response = str(addr).zfill(5) + " : " + "CLIENT DEAD"
                    else:
                        response = c.read_input_registers(int(addr), 1)
                        response = str(addr).zfill(5) + " : " + str(response)
                    try:
                        stdscr.addstr(start_y, start_x, str(response))
                        start_y += 1
                    except Exception as e:
                        print(e)
                        print("Increase the size of terminal window!")

        stdscr.refresh()

        k = stdscr.getch()
        if k == ord('s'):
            if server_started:
                server_status = "server running!"
            else:
                bash = "python modbus/modbus_server.py -address {0}".format(args.address)
                server = subprocess.Popen(bash.split(), stdout=subprocess.PIPE)
                server_started = True
                status = "SERVER STARTING"
        elif k == ord('k'):
            if server_started:
                server.terminate()
#                outps, error = server.communicate()
                server_started = False
                status = "SERVER KILLED"
            else:
                server_status = "Haven't started server"
        elif k == ord('c'):
            if client_started:
                client_status = "client running!"
            else:
                clients = []
                for i in range(len(stream_txt)):
                    client_shift = stream_index[i].split('-')[0]
                    bash = "python counter.py --slave-id {0} --source {1} --address {2} --client-shift {3}".format(args.i,stream_txt[i],args.address, client_shift)
                    clients.append(subprocess.Popen(bash.split(), stdout=subprocess.DEVNULL))
                timeout = time.time()
                client_started = True
                client_status = "CLIENT RUNNING"
                status = "CLIENT STARTING"
        elif k == ord('d'):
            if client_started:
                for i in range(len(stream_txt)):
                    try:
                        clients[i].kill()
                    except:
                        continue
#                outps, error = server.communicate()
                client_started = False
                client_status = "CLIENT NOT RUNNING"
                status = "CLIENT KILLED"
            else:
                client_status = "Haven't start client"
        elif k == ord('t'):
            if server_started:
                server_status = "server running!"
            else:
                bash = "python modbus/modbus_server.py -address {0}".format(args.address)
                server = subprocess.Popen(bash.split(), stdout=subprocess.PIPE)
                server_started = True
                status = "SERVER STARTING"
            if client_started:
                client_status = "client running!"
            else:
                clients = []
                for i in range(len(stream_txt)):
                    client_shift = stream_index[i].split('-')[0]
                    bash = "python counter.py --slave-id {0} --source {1} --address {2} --client-shift {3}".format(args.i,stream_txt[i],args.address, client_shift)
                    clients.append(subprocess.Popen(bash.split(), stdout=subprocess.DEVNULL))
                timeout = time.time()
                client_started = True
                client_status = "CLIENT RUNNING"
                status = "CLIENT STARTING"
        elif k == ord('y'):
            if server_started:
                server.terminate()
#                outps, error = server.communicate()
                server_started = False
                status = "SERVER KILLED"
            else:
                server_status = "Server is not managed in this client"
            if client_started:
                for i in range(len(stream_txt)):
                    try:
                        clients[i].kill()
                    except:
                        continue
                client_started = False
                client_status = "CLIENT NOT RUNNING"
                status = "CLIENT KILLED"
            else:
                client_status = "Haven't start client"
        elif k == ord('q'):
            if server_started:
                server.kill()
            if client_started:
                for i in range(len(stream_txt)):
                    try:
                        clients[i].kill()
                    except:
                        continue
            break

        if client_started:
            for i in range(len(stream_txt)):
                if clients[i].poll() is not None:
                    client_status  = "SOME CLIENTS DIED"
#                        client_started = False
                    status = "RUNNING"
                else:
                    client_status = "CLIENT RUNNING"
                    if not args.no_timeout:
#                        if ((time.time() - timeout) > (3600*3)):
                        if ((time.time() - timeout) > (60*10)):
                            for i in range(len(stream_txt)):
                                try:
                                    clients[i].kill()
                                except:
                                    continue
                            client_started = False
                            client_status = "CLIENT NOT RUNNING"
                            status = "CLIENT RESTARTING DUE TO TIMEOUT"
                            initial = True
        else:
            client_status  = "CLIENT NOT RUNNING"
            status = "CLIENT IS DEAD"
            if args.auto_restart:
                clients = []
                for i in range(len(stream_txt)):
                    client_shift = stream_index[i].split('-')[0]
                    bash = "python counter.py --slave-id {0} --source {1} --address {2} --client-shift {3}".format(args.i,stream_txt[i],args.address, client_shift)
                    clients.append(subprocess.Popen(bash.split(), stdout=subprocess.DEVNULL))
                timeout = time.time()
                client_started = True
                client_status = "CLIENT RUNNING"
                status = "CLIENT STARTING"
        if server_started:
            if server.poll() is not None:
                server_status  = "SERVER NOT RUNNING"
                server_started = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-address', type=str, default='localhost', help='server address')
    parser.add_argument('-i', type=int, default=1, help='slave id')
    parser.add_argument('-sf', '--stream-file', default='stream.txt', type=str, help='list of streams to open')
    parser.add_argument('-auto', '--auto-restart', action='store_true', help='autorestart client')
    parser.add_argument('-notime', '--no-timeout', action='store_true', help='no timeout')
    args = parser.parse_args()
    stream_txt = []
    stream_index = []
    f = open(args.stream_file, "r")
    s = f.read().strip()
    s = s.split("\n")
    for i in s[:]:
        n,v = i.split(" ")
        stream_txt.append(v)
        stream_index.append(n)
    f.close()
    try:
        curses.wrapper(main)
    except Exception as e:
        print(e)
