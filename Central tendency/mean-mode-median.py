import statistics
def cal_mean(data):
  return sum(data)/len(data)
def cal_median(data):
  n=len(data)
  if(n%2==0):
    m1=data[n//2]
    m2=data[(n//2)+1]
    return (m1+m2)/2
  else:
    return data[n//2]
def cal_mode(data):
  return statistics.mode(data)
def cal_varience(data):
  mean=cal_mean(data)
  varience=sum((x-mean)**2 for x in data)/len(data)
  return varience
def cal_SD(data):
  varience=cal_varience
  SD=statistics.stdev(data)
  return SD
data=[10,20,30,40,50]
print('Mean:',cal_mean(data))
print('Median:',cal_median(data))
print('Mode:',cal_mode(data))
print('Varience:',cal_varience(data))
print('Standard Deviation:',cal_SD(data))
     