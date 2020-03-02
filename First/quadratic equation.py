#利用Python完成一元二次方程ax^2+bx+c=0的求解，要求程序输入任意a,b,c的值后，程序能判断输出有解或无解，有解的话，输出x的值为多少。
#author：王磊
#Sno：2017051603016

import math

def cacul(a, b, c):
    if a != 0:
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            print("无根")
        elif delta == 0:
            s = -b/(2 * a)
            print("唯一根x = ",s)
        else :
            root = math.sqrt(delta)
            x1 = (-b + root) / (2 * a)
            x2 = (-b - root) / (2 * a)
            print("x1 = ", x1, "\n", "x2 = ", x2)

def main():
    print("利用Python完成一元二次方程ax^2+bx+c=0的求解")
    print("--"*20)
    a = float(input("请输入a的值："))
    b = float(input("请输入b的值："))
    c = float(input("请输入c的值："))
    cacul(a, b, c)

if __name__ == '__main__':
    main()