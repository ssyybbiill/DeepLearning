print("My Name is %s and weight is %d kg" % ('hxl', 20))
s = 'a,B,c'
print(s.capitalize())
s = ' a bc,bc,aaabcb'  # 第一个是空格，没有大写
print(s.capitalize())
print(s.center(20, '*'))  # width -- 字符串的总宽度，fillchar -- 填充字符。
print(s.count('bc', 4, 14))
