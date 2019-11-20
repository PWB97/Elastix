from helper import *


finalRes = []
# for i in range(5):
getMovingIMG()
elastix()

res = []
for dir in os.listdir(OUTPUT):
    try:
        res.append(getTRE(dir))
    except Exception:
        print(dir)

print('-------------------------------------')
# print('time no.',i)
print(res)
res_mean = np.array(res).mean()
print(res_mean)
print('-------------------------------------')
finalRes.append(res_mean)


# print('final')
# print(finalRes)
# print(np.array(finalRes).mean())
#
# line = "********-" + 'TRE' + '-********-[TRE mean, TRE min, TRE max, TRE std, TRE median, 90% percentile]: '
# print(line)
# f = open(OUTPUT + 'tre.txt', 'w+')
# f.write(line + '\n')
# result = np.mean(finalRes), np.min(finalRes), np.max(finalRes), np.std(finalRes), np.median(finalRes), np.percentile(finalRes, 90)
# f.write(str(result[0])+'    ')
# f.write(str(result[1])+'    ')
# f.write(str(result[2])+'    ')
# f.write(str(result[3])+'    ')
# f.write(str(result[4])+'    ')
# f.write(str(result[5]))