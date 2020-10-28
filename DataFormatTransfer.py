import math

def Decimal_to_AnyHex(Data, target_Hex):
    '''this is used to transfer decimal to any hex'''
    result = []
    temp = 0
    while Data / target_Hex != 0:
        rest = Data % target_Hex
        result.append(rest)
        Data = int(Data / target_Hex)
    num = len(result)
    for i in range(num):
        temp += result.pop() * math.pow(10, num - 1)
        num = num - 1
    return int(temp)


def AnyHex_to_Decimal(Data, target_Hex):
    '''this is transfer any hex to decimal'''
    string = str(Data)
    length = len(string)
    result = 0
    map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, \
           "A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15}
    for index, i in enumerate(string):
        power_y = length - index - 1
        power_x = map[str(i)]
        result += power_x * math.pow(target_Hex, power_y)
    # print(int(result))
    return int(result)


if __name__ == '__main__':
    # print(Hex_to_D('49570'))
    print(Decimal_to_AnyHex(150, 11))
    print(AnyHex_to_Decimal(Decimal_to_AnyHex(150, 11), 11))
