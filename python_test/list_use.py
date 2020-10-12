list1 = ['a', 'ddddd', 234]
print(list1)
print(len(list1))
print(list1[0:2])  # 第一个开始，共两个
list1.append('aaa')
print(list1)
del list1[1]
print(list1)
list1 += [1, 3, 234]
print(list1)
print(list1 * 3)
if (1 in list1):
    print("1在里面")
else:
    print("1不在里面")
for x in list1:
    print(x)
list1 = [1, 3, -2]
print(max(list1))
list1.extend([3, 3, 3, 5])
print(list1)
list1.append([3, 3, 3, 5])
print(list1)
print(list1.pop(-1))
print(list1)
list1.reverse()
print(list1)
list1.sort(reverse=True)
print(list1)


def takeSecond(elem):
    return elem[1]


list2 = [(2, 3), (2, 4), (7, 2), (6, -1)]
list2.sort(key=takeSecond)
print(list2)
