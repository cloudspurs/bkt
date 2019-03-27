a = b = c = d = 0

for i in range(len(ss)-1):
    if ss[i] == 0:
        # 0, 0
        if ss[i+1] == ss[i]:
            a += 1
        # 0, 1
        if ss[i+1] > ss[i]:
            b += 1
    if ss[i] == 1:
        # 1, 0
        if ss[i+1] < ss[i]:
            c += 1
        # 1, 1
        if ss[i+1] == ss[i]:
            d+=1
print('\n', a, b, c, d)
print("0 -> 0: ", a/(a+b))
print("0 -> 1: ", b/(a+b))
print("1 -> 0: ", c/(c+d))
print("1 -> 1: ", d/(c+d))

e = f = g = h = 0

for i in range(len(ss)):
    if ss[i] == 0:
        if os[i] == 0:
            e += 1
        if os[i] == 1:
            f += 1
    if ss[i] == 1:
        if os[i] == 0:
            g += 1
        if os[i] == 1:
            h += 1

print('\n', e, f, g, h)
print("0 -> 0: ", e/(e+f))
print("0 -> 1: ", f/(e+f))
print("1 -> 0: ", g/(g+h))
print("1 -> 1: ", h/(g+h))
