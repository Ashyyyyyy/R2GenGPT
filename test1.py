
#
# a = "abcde"
# a= list(a)
# print(a)
# print(a[0], a[1], a[2], a[3], a[4])
# print(f'{a[0]}:{a[1]}:{a[2]}:{a[3]}:{a[4]}')
# print(a[0]+":"+a[1]+":"+a[2]+":"+a[3]+":"+a[4])
# # a = ''.join(a)
# # print(a)

num11 = input()
num1 = num11.split(',')
num22 = input()
num2 = num22.split(',')
n = len(num1)
num1[n:]= num2
print(num1)

# a = input()
# print(a)