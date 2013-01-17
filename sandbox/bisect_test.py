import bisect

a = [1,3,5,5,7,9]
x = bisect.bisect_left(a, 6, 2, 4)
print x
x = bisect.bisect_left(a, 7, 2, 4)
print x
x = bisect.bisect_left(a, 5, 2, 4)
print x
