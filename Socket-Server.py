# -*- coding: utf-8 -*-



from PyQt5.QtWidgets import QApplication, QMainWindow
from Dialog import Ui_Dialog
from PyQt5.QtCore import QThread, pyqtSignal
import socket
import time
import sys


class TcpServerThread(QThread):
    server_msg_out = pyqtSignal(str)
    def __init__(self,):
        super(TcpServerThread, self).__init__()
        self.client_socket_list = list()

    def start_listen(self,ip,port):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_socket.setblocking(False)

        try:
            server_ip = str(ip)
            server_port = int(port)
            print(f'success get the ip and address {ip}, {port}')
            self.tcp_socket.bind((server_ip, server_port))
        except Exception as e:
            msg = '请检查IP 和 端口号 \n'
            self.server_msg_out.emit(str(msg))
        else:
            print(f'start listen from {ip}, {port}')
            self.tcp_socket.listen()
            # self.server_thread = threading.Thread(target=self.tcp_server_concurrency)
            # self.server_thread.start()
            msg = f'TCP服务器端正在监听: {server_ip, server_port} \n'
            self.server_msg_out.emit(msg)

    def run(self):
        """
               功能函数，供创建线程的方法；
               使用子线程用于监听并创建连接，使主线程可以继续运行，以免无响应
               使用非阻塞式并发用于接收客户端消息，减少系统资源浪费，使软件轻量化
               :return:None
               """
        while True:
            try:
                client_socket, client_address = self.tcp_socket.accept()
            except Exception as e:
                time.sleep(0.001)
            else:
                client_socket.setblocking(False)
                self.client_socket_list.append((client_socket, client_address))
                msg = f'TCP服务器端已经连接 {client_address} \n'
                self.server_msg_out.emit(msg)
            for client_socket, client_address in self.client_socket_list:
                try:
                    recv_msg = client_socket.recv(1024)
                except Exception as e:
                    pass
                else:
                    if recv_msg:
                        msg = recv_msg.decode('utf-8')
                        msg = f'信息来自: {client_address} :  {msg} \n'
                        self.server_msg_out.emit(msg)
                        client_socket.send('Data Received\n'.encode(encoding='utf-8'))
                    else:
                        client_socket.close()
                        self.client_socket_list.remove((client_socket, client_address))


class TcpClientThread(QThread):
    client_msg_out = pyqtSignal(str)
    def __init__(self, ):
        super(TcpClientThread, self).__init__()
        self.num = 0
    def start_connnect(self, ip, port):
        self.client_tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)
        self.client_tcp_socket.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 60 * 1000, 30 * 1000))
        self.client_tcp_socket.settimeout(1000)

        try:
            server_ip = str(ip)
            server_port = int(port)
            address = (server_ip, server_port)
        except Exception as e:
            msg = '请检查服务器IP 和端口\n'
            self.client_msg_out.emit(msg)
        else:
            try:
                msg = '正在连接服务器\n'
                self.client_msg_out.emit(msg)
                self.client_tcp_socket.connect(address)
            except Exception as e:
                msg = '无法连接目标服务器 \n'
                self.client_msg_out.emit(msg)
            else:
                # self.client_thread = threading.Thread(target=self.tcp_client_concurrency, args=(address,))
                # self.client_thread.start()
                msg = f'TCP 客户端已经连接IP : {address} \n'
                self.client_msg_out.emit(msg)
    def send_data(self, msg):
        if msg=='':
            msg = 'input empty！'
        self.client_tcp_socket.send(msg.encode('utf-8'))

    def run(self):
        while True:
            recv_msg = self.client_tcp_socket.recv(1024)
            if recv_msg:
                msg = recv_msg.decode('utf-8')
                self.client_msg_out.emit(msg)
            else:
                self.client_tcp_socket.close()
                # self.reset()
                msg = '从服务器断开!\n'
                self.client_msg_out.emit(msg)
                break


class MainWindow_New(QMainWindow, Ui_Dialog):
    def __init__(self,  ):
        super(MainWindow_New,self).__init__()
        self.client_socket_list = list()
        self.setupUi(self)
        self.server = TcpServerThread()
        self.client = TcpClientThread()
        self.connect()


    def connect(self):
        # super(MainWindow_New,self).connect()
        self.Btn_Server_Start_Listen.clicked.connect(self.click_server_start)
        self.Btn_Client_Send.clicked.connect(self.click_client_send)
        self.Btn_Server_Clear.clicked.connect(self.clear)
        self.Btn_Client_Clear.clicked.connect(self.client_clear)
        self.Btn_Client_Connet_Server.clicked.connect(self.click_client_connect)
        self.server.server_msg_out.connect(self.write_server_msg)
        self.client.client_msg_out.connect(self.write_client_msg)

    def click_server_start(self):

        ip = self.Server_IP_Edit.text()
        port = self.Server_Port_Edit.text()
        self.server.start_listen(ip=ip, port=port)
        self.server.start()
        # self.server.server_msg_out.connect(self.write_server_msg)

    def write_server_msg(self,str):
        self.Server_Received_Text.insertPlainText(str)

    def write_client_msg(self, str):
        self.Client_Connect_Status.insertPlainText(str)

    def click_client_connect(self):
        ip = self.Client_IP_Edit.text()
        port = self.Client_Port_Edit.text()
        self.client.start_connnect(ip=ip, port=port)
        self.client.start()

    def click_client_send(self):
        msg = self.Text_From_Client.toPlainText()
        self.client.send_data(msg)

    def clear(self):
        self.Server_Received_Text.clear()
    def client_clear(self):
        self.Client_Connect_Status.clear()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ui = MainWindow_New()
    ui.show()
    sys.exit(app.exec_())


