import serial
import time
import json
from collections import deque, Counter

class SerialCommunicator:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=9600, timeout=1):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn = None
        self.connect()

    def connect(self):
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            time.sleep(2)  # 等待串口连接稳定
            print(f'串口 {self.serial_port} 已打开，波特率: {self.baud_rate}')
        except serial.SerialException as e:
            print(f'无法打开串口 {self.serial_port}: {e}')
            self.serial_conn = None

    def send_data(self, person_label):
        buffer_size = 30
        buffer_mode = BufferWithMode(buffer_size)
        buffer_out = buffer_mode.add_item(person_label)
        if self.serial_conn and self.serial_conn.is_open:
            try:
                # 初始化数据为0
                data_byte = 0x00

                # 根据识别结果设置对应的位
                if buffer_out == 'person1':
                    data_byte |= 0x08  # 设置Bit 0
                elif buffer_out == 'person2':
                    data_byte |= 0x04  # 设置Bit 1
                elif buffer_out == 'person3':
                    data_byte |= 0x02  # 设置Bit 2
                elif buffer_out == 'unknown':
                    data_byte |= 0x01  # 设置Bit 3

                # 高4位补0，已经是0x00

                # 将数据转换为字节类型发送
                self.serial_conn.write(bytes([data_byte]))
                print(f'发送串口数据: {data_byte:08b}')  # 以二进制形式打印发送的数据
            except serial.SerialException as e:
                print(f'串口发送失败: {e}')
        else:
            print('串口未连接或已关闭，无法发送数据')

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print('串口连接已关闭')

class BufferWithMode:
    def __init__(self, size):
        """
        初始化缓冲区。
        
        :param size: 缓冲区的最大容量
        """
        self.size = size
        self.buffer = deque(maxlen=size)
        self.counter = Counter()
    
    def add_item(self, number):
        """
        添加一个新的数字到缓冲区，并更新计数器。
        
        :param number: 要添加的数字
        :return: 当前缓冲区的众数
        """
        if len(self.buffer) == self.size:
            # 缓冲区已满，移除最旧的元素
            oldest = self.buffer.popleft()
            self.counter[oldest] -= 1
            if self.counter[oldest] == 0:
                del self.counter[oldest]
        
        # 添加新元素
        self.buffer.append(number)
        self.counter[number] += 1
        
        # 计算众数
        mode = self.get_mode()
        return mode
    
    def get_mode(self):
        """
        获取当前缓冲区的众数。如果有多个众数，返回其中一个。
        
        :return: 众数
        """
        if not self.counter:
            return None
        # 找到最高的出现次数
        max_count = max(self.counter.values())
        # 返回所有出现次数等于最高次数的数字中的第一个
        for num in self.buffer:
            if self.counter[num] == max_count:
                return num

